#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd
from math import sqrt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, norm

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 프로젝트 모듈 import
from cms_pricing.src.market import bootstrap_if_needed, create_continuous_zero_curve
from cms_pricing.src.models.calibration import (
    build_market_prices_from_vol_surface,
    calculate_forward_swap_rate_market,
    price_european_swaption_g2_fast,
    black_swaption_price,
)
from cms_pricing.src.models import load_calibrated_params
from cms_pricing.config.settings import EXPIRY_LABELS, TENORS, DATA_DIR, VOLATILITY_SURFACE_FILE # Import DATA_DIR and VOLATILITY_SURFACE_FILE

def main() -> None:
    """G2++와 Black, Bachelier 모델의 적합도를 비교하여 금리 모델을 평가합니다."""
    # 1. 시장 데이터와 할인곡선 로드
    par_full, P_year = bootstrap_if_needed()
    zero_curve = create_continuous_zero_curve(P_year)
    P_market = lambda t: np.exp(-zero_curve(t) * t)

    # 2. 변동성 표면 로드 (리스케일된 vol을 % 단위로 환산)
    volatility_surface_path = os.path.join(DATA_DIR, VOLATILITY_SURFACE_FILE)
    with open(volatility_surface_path) as f:
        surf = json.load(f)
    vol_rescaled = pd.DataFrame(surf['values'], index=surf['index'], columns=surf['columns'])
    surface_pct = vol_rescaled.values * 100.0

    # 3. 시장 ATM 스왑션 가격 계산
    market_prices = build_market_prices_from_vol_surface(surface_pct, EXPIRY_LABELS, TENORS, P_market)
    market_vals = np.array(list(market_prices.values()))
    keys = list(market_prices.keys())
    # 각 만기·테너에 대한 forward rate와 annuity 저장
    forward_annuity = [calculate_forward_swap_rate_market(T, tenor, P_market) for T, tenor in keys]

    # -------------------------------------------------------------------------
    # G2++ 모델 가격 및 지표
    # -------------------------------------------------------------------------
    params = load_calibrated_params()
    model_prices = {}
    for (T, tenor) in keys:
        fwd, ann = calculate_forward_swap_rate_market(T, tenor, P_market)
        # ATM이라 strike = forward
        model_prices[(T, tenor)] = price_european_swaption_g2_fast(
            params, (T, tenor, fwd, 1.0), P_market
        )
    model_vals = np.array([model_prices[k] for k in keys])
    # R^2 및 RMSE 계산
    rss = np.sum((market_vals - model_vals) ** 2)
    tss = np.sum((market_vals - np.mean(market_vals)) ** 2)
    r2_g2 = 1.0 - rss / tss
    rmse_g2 = np.sqrt(np.mean((market_vals - model_vals) ** 2))
    # 단순 회귀 R^2와 p-value
    slope, intercept, r_value, p_value, _ = linregress(market_vals, model_vals)
    r2_g2_lin = r_value ** 2
    p_g2 = p_value

    # -------------------------------------------------------------------------
    # GBM(Black) 모델: 상수 변동성 추정
    # -------------------------------------------------------------------------
    def black_objective(vol):
        preds = []
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys):
            # ATM이므로 strike = forward
            preds.append(black_swaption_price(fwd, fwd, vol, T, ann, True))
        preds = np.array(preds)
        return np.sum((preds - market_vals) ** 2)

    res_black = minimize_scalar(
        black_objective, bounds=(1e-4, 1.0), method='bounded', options={'xatol': 1e-6}
    )
    vol_black = res_black.x
    # 상수 변동성으로 예측 가격 계산
    pred_black = np.array([
        black_swaption_price(fwd, fwd, vol_black, T, ann, True)
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys)
    ])
    rss_b = np.sum((market_vals - pred_black) ** 2)
    r2_black = 1.0 - rss_b / tss
    rmse_black = np.sqrt(np.mean((market_vals - pred_black) ** 2))
    slope_b, intercept_b, r_b, p_b, _ = linregress(market_vals, pred_black)
    r2_black_lin = r_b ** 2

    # -------------------------------------------------------------------------
    # Bachelier(정규) 모델: 상수 정규 변동성 추정
    # -------------------------------------------------------------------------
    def bachelier_swaption_price(fwd, strike, vol, expiry, annuity):
        if expiry <= 0 or vol <= 0:
            return 0.0
        d = (fwd - strike) / (vol * sqrt(expiry))
        return annuity * ((fwd - strike) * norm.cdf(d) + vol * sqrt(expiry) * norm.pdf(d))

    def bach_objective(vol):
        preds = []
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys):
            preds.append(bachelier_swaption_price(fwd, fwd, vol, T, ann))
        preds = np.array(preds)
        return np.sum((preds - market_vals) ** 2)

    res_bach = minimize_scalar(
        bach_objective, bounds=(1e-4, 1.0), method='bounded', options={'xatol': 1e-6}
    )
    vol_bach = res_bach.x
    pred_bach = np.array([
        bachelier_swaption_price(fwd, fwd, vol_bach, T, ann)
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys)
    ])
    rss_bn = np.sum((market_vals - pred_bach) ** 2)
    r2_bach = 1.0 - rss_bn / tss
    rmse_bach = np.sqrt(np.mean((market_vals - pred_bach) ** 2))
    slope_bn, intercept_bn, r_bn, p_bn, _ = linregress(market_vals, pred_bach)
    r2_bach_lin = r_bn ** 2

    # -------------------------------------------------------------------------
    # 결과 출력
    # -------------------------------------------------------------------------
    print("=== 적합도 비교 (ATM 스왑션) ===")
    print(f"G2++ 모델:      R^2={r2_g2:.4f}, RMSE={rmse_g2:.6f}, 회귀 R^2={r2_g2_lin:.4f}, p-value={p_g2:.2e}")
    print(f"Black(GBM):     R^2={r2_black:.4f}, RMSE={rmse_black:.6f}, 회귀 R^2={r2_black_lin:.4f}, p-value={p_b:.2e}, 최적 σ={vol_black:.4f}")
    print(f"Bachelier:      R^2={r2_bach:.4f}, RMSE={rmse_bach:.6f}, 회귀 R^2={r2_bach_lin:.4f}, p-value={p_bn:.2e}, 최적 σ_N={vol_bach:.4f}")

if __name__ == "__main__":
    main()