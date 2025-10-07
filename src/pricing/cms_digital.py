# -*- coding: utf-8 -*-
"""
디지털 CMS 스프레드 노트 가격결정 모듈
======================================

이 모듈은 G2++ 모델과 φ(t) 함수를 이용하여 CMS 스프레드 기반
디지털 노트의 가격을 계산합니다. 시뮬레이션 경로를 생성하고,
지급 조건에 따라 할인된 페이오프의 평균을 구하는 방식으로 가격을
산출합니다.
"""

from typing import Callable, Dict, Tuple
import numpy as np

from ..models.g2pp import calculate_forward_swap_rate
from ..pricing.simulation import simulate_paths


def price_digital_cms_spread(params: Dict[str, float],
                             product_details: Tuple[float, float, float, float, float, float],
                             P_market: Callable[[float], float],
                             V_0_func: Callable[[float], float],
                             phi_func: Callable[[float], float],
                             num_paths: int = 10000) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    """디지털 CMS 스프레드 노트의 가격을 계산합니다.

    product_details = (만기T, tenor_long, tenor_short, strike, coupon, notional)

    반환값은 (가격, 스프레드 배열, 할인계수 배열, 통계 딕셔너리)입니다.
    """
    T_expiry, tenor_long, tenor_short, strike, coupon, notional = product_details
    # 시뮬레이션 스텝 수는 영업일 기준 약 252일을 가정
    num_steps = max(int(T_expiry * 252), 100)
    # 경로 생성
    x_T, y_T, D_paths = simulate_paths(params, T_expiry, num_paths, num_steps, phi_func)
    # 할인계수 검증: E[D_T] ≈ P(0,T)
    df_T = P_market(T_expiry)
    mean_D = float(np.mean(D_paths))
    error_pct = abs(mean_D - df_T) / df_T * 100.0 if df_T > 0 else 0.0
    # CMS 스프레드 계산
    spreads = np.zeros(num_paths, dtype=float)
    for i in range(num_paths):
        rate_long = calculate_forward_swap_rate(T_expiry, tenor_long, x_T[i], y_T[i], params, P_market, V_0_func)
        rate_short = calculate_forward_swap_rate(T_expiry, tenor_short, x_T[i], y_T[i], params, P_market, V_0_func)
        spreads[i] = rate_long - rate_short
    # 디지털 페이오프
    indicators = (spreads > strike).astype(float)
    payoffs = coupon * indicators
    price_per_unit = float(np.mean(D_paths * payoffs))
    price = price_per_unit * notional
    E_D1 = float(np.mean(D_paths * indicators))
    payout_prob = float(np.mean(indicators))
    p_T = E_D1 / df_T if df_T > 0 else 0.0
    fair_coupon = (1.0 / E_D1) if E_D1 > 0 else float('inf')
    stats = {
        "mean_D": mean_D,
        "df_T": df_T,
        "payout_prob": payout_prob,
        "p_T": p_T,
        "E_D1": E_D1,
        "fair_coupon": fair_coupon,
        "spread_mean": float(np.mean(spreads)),
        "spread_std": float(np.std(spreads)),
        "discount_error_pct": error_pct,
    }
    return price, spreads, D_paths, stats