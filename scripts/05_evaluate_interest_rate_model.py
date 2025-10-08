#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cms_pricing/scripts/05_evaluate_interest_rate_model.py

import os
import sys
import json
import numpy as np
import pandas as pd
from math import sqrt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, norm

# add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# project imports
from cms_pricing.src.market import bootstrap_if_needed, create_continuous_zero_curve
from cms_pricing.src.models.calibration import (
    build_market_prices_from_vol_surface,
    calculate_forward_swap_rate_market,
    price_european_swaption_g2_fast,
    black_swaption_price,   # signature: (fwd, strike, vol, expiry, annuity)
)
from cms_pricing.src.models import load_calibrated_params
from cms_pricing.config.settings import EXPIRY_LABELS, TENORS, DATA_DIR, VOLATILITY_SURFACE_FILE


def main() -> None:
    """Evaluate interest rate models (G2++, Black, Bachelier) against ATM swaption prices."""

    # 1) market data & discount curve
    par_full, P_year = bootstrap_if_needed()
    zero_curve = create_continuous_zero_curve(P_year)
    P_market = lambda t: np.exp(-zero_curve(t) * t)

    # 2) load rescaled vol surface (as %) and convert to list of lists
    vol_path = os.path.join(DATA_DIR, VOLATILITY_SURFACE_FILE)
    with open(vol_path, "r", encoding="utf-8") as f:
        surf = json.load(f)
    vol_rescaled = pd.DataFrame(surf["values"], index=surf["index"], columns=surf["columns"])
    surface_pct = vol_rescaled.values.astype(float) * 100.0
    surface_pct = surface_pct.tolist()  # ensure list-of-lists

    # 3) market ATM swaption prices via Black (ATM strike = forward)
    # NOTE: calibration.build_market_prices_from_vol_surface signature:
    #   (p_market_func, surface_pct, expiry_labels, tenors)
    market_prices = build_market_prices_from_vol_surface(P_market, surface_pct, EXPIRY_LABELS, TENORS)
    keys = list(market_prices.keys())
    market_vals = np.array([market_prices[k] for k in keys], dtype=float)

    # precompute (fwd, annuity) for each instrument key
    forward_annuity = [calculate_forward_swap_rate_market(T, tenor, P_market) for (T, tenor) in keys]

    # ---------------------------------------------------------------------
    # G2++ model
    # ---------------------------------------------------------------------
    params = load_calibrated_params()  # {'a','b','sigma','eta','rho',...}

    model_prices = {}
    for (T, tenor) in keys:
        fwd, ann = calculate_forward_swap_rate_market(T, tenor, P_market)
        # price_european_swaption_g2_fast signature:
        #   (params, p_market_func, expiry, tenor, strike)
        model_prices[(T, tenor)] = price_european_swaption_g2_fast(params, P_market, T, tenor, fwd)

    model_vals = np.array([model_prices[k] for k in keys], dtype=float)

    rss = np.sum((market_vals - model_vals) ** 2)
    tss = np.sum((market_vals - np.mean(market_vals)) ** 2) + 1e-16
    r2_g2 = 1.0 - rss / tss
    rmse_g2 = float(np.sqrt(np.mean((market_vals - model_vals) ** 2)))

    slope, intercept, r_value, p_value, _ = linregress(market_vals, model_vals)
    r2_g2_lin = r_value ** 2
    p_g2 = p_value

    # ---------------------------------------------------------------------
    # Black (GBM) model: single constant vol fit
    # ---------------------------------------------------------------------
    def black_objective(vol):
        preds = []
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys):
            # NOTE: black_swaption_price signature = (fwd, strike, vol, expiry, annuity)
            preds.append(black_swaption_price(fwd, fwd, vol, T, ann))
        preds = np.array(preds, dtype=float)
        return np.sum((preds - market_vals) ** 2)

    res_black = minimize_scalar(
        black_objective, bounds=(1e-4, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    vol_black = float(res_black.x)

    pred_black = np.array([
        black_swaption_price(fwd, fwd, vol_black, T, ann)
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys)
    ], dtype=float)
    rss_b = np.sum((market_vals - pred_black) ** 2)
    r2_black = 1.0 - rss_b / tss
    rmse_black = float(np.sqrt(np.mean((market_vals - pred_black) ** 2)))
    slope_b, intercept_b, r_b, p_b, _ = linregress(market_vals, pred_black)
    r2_black_lin = r_b ** 2

    # ---------------------------------------------------------------------
    # Bachelier (normal) model: single constant normal vol fit
    # ---------------------------------------------------------------------
    def bachelier_swaption_price(fwd, strike, vol, expiry, annuity):
        if expiry <= 0 or vol <= 0 or annuity <= 0:
            return 0.0
        d = (fwd - strike) / (vol * sqrt(expiry))
        return annuity * ((fwd - strike) * norm.cdf(d) + vol * sqrt(expiry) * norm.pdf(d))

    def bach_objective(vol):
        preds = []
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys):
            preds.append(bachelier_swaption_price(fwd, fwd, vol, T, ann))
        preds = np.array(preds, dtype=float)
        return np.sum((preds - market_vals) ** 2)

    res_bach = minimize_scalar(
        bach_objective, bounds=(1e-4, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    vol_bach = float(res_bach.x)

    pred_bach = np.array([
        bachelier_swaption_price(fwd, fwd, vol_bach, T, ann)
        for (fwd, ann), (T, tenor) in zip(forward_annuity, keys)
    ], dtype=float)
    rss_bn = np.sum((market_vals - pred_bach) ** 2)
    r2_bach = 1.0 - rss_bn / tss
    rmse_bach = float(np.sqrt(np.mean((market_vals - pred_bach) ** 2)))
    slope_bn, intercept_bn, r_bn, p_bn, _ = linregress(market_vals, pred_bach)
    r2_bach_lin = r_bn ** 2

    # ---------------------------------------------------------------------
    # results
    # ---------------------------------------------------------------------
    print("=== Goodness-of-fit (ATM swaptions) ===")
    print(f"G2++:          R^2={r2_g2:.4f}, RMSE={rmse_g2:.6f}, linear R^2={r2_g2_lin:.4f}, p-value={p_g2:.2e}")
    print(f"Black (GBM):   R^2={r2_black:.4f}, RMSE={rmse_black:.6f}, linear R^2={r2_black_lin:.4f}, p-value={p_b:.2e}, best σ={vol_black:.4f}")
    print(f"Bachelier:     R^2={r2_bach:.4f}, RMSE={rmse_bach:.6f}, linear R^2={r2_bach_lin:.4f}, p-value={p_bn:.2e}, best σ_N={vol_bach:.4f}")


if __name__ == "__main__":
    main()
