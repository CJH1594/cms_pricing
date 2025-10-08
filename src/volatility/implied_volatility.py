#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..models.analytic import GBMModel, BachelierModel # Import the models

# Create model instances
gbm_model_instance = GBMModel()
bachelier_model_instance = BachelierModel()

def calculate_implied_volatility_small_root_gbm(S: float, K: float, T: float,
                                                target_price: float) -> tuple[float, bool]:
    """
    [GBM] 디지털 콜 옵션의 작은 해 내재변동성 역산.

    f(σ) = target_price - P(σ) 를 정의하면, σ=0 부근에서 양수, σ=vol_peak 부근에서 음수인 경우
    [0, vol_peak] 구간에 해가 존재한다. 그 해를 찾아 반환한다.

    Returns: (implied_vol, converged)
    """
    if S <= 0 or K <= 0 or T <= 0:
        return 0.0, False

    model_price = lambda vol: gbm_model_instance.price_digital(S, K, T, vol)
    objective = lambda vol: target_price - model_price(vol)

    if S == K:
        # ATM에서는 가격이 0.5 근처가 아니면 해가 없음
        return (0.20, True) if np.isclose(target_price, 0.5) else (0.0, False)

    vol_peak = np.sqrt(max(0.0, 2.0 * abs(np.log(S / K)) / T))
    vol_min, vol_max = 1e-8, max(vol_peak, 1e-6)

    try:
        f_min = objective(vol_min)
        f_max = objective(vol_max)
        if f_min * f_max > 0:
            return 0.0, False
        vol = brentq(objective, vol_min, vol_max, xtol=1e-12, rtol=1e-12)
        return vol, True
    except (ValueError, RuntimeError):
        return 0.0, False


def calculate_implied_volatility_small_root_bachelier(S: float, K: float, T: float,
                                                     target_price: float) -> tuple[float, bool]:
    """
    [Bachelier] 디지털 콜 옵션의 내재변동성 역산.

    Bachelier 모델 가격은 변동성에 대해 단조 함수이므로 해는 유일하다.
    [0, large_vol] 구간에서 해를 찾는다.

    Returns: (implied_vol, converged)
    """
    if T <= 0:
        return 0.0, False

    # ATM에서는 가격이 0.5 근처가 아니면 해가 없음
    if S == K:
        return (0.005, True) if np.isclose(target_price, 0.5) else (0.0, False)

    model_price = lambda vol: bachelier_model_instance.price_digital(S, K, T, vol)
    objective = lambda vol: target_price - model_price(vol)

    # Bachelier 모델은 단조 함수이므로 넓은 구간에서 탐색
    vol_min, vol_max = 1e-8, 5.0  # 500% vol

    try:
        f_min = objective(vol_min)
        f_max = objective(vol_max)
        if f_min * f_max > 0:
            return 0.0, False # 해가 구간 내에 없음
        vol = brentq(objective, vol_min, vol_max, xtol=1e-12, rtol=1e-12)
        return vol, True
    except (ValueError, RuntimeError):
        return 0.0, False


def calculate_implied_volatility(pricing_result: dict, use_bachelier: bool = True) -> float:
    """
    디지털 CMS 스프레드 노트의 내재변동성을 역산.

    pricing_result: { ... }
    use_bachelier: True이면 Bachelier, False이면 GBM 모델 사용
    """
    prod = pricing_result['product']
    S, K, T = prod['spread_asset_price'], prod['strike'], prod['expiry']
    notional, coupon = prod['notional'], prod['coupon']
    note_value = pricing_result['price']

    if abs(coupon) < 1e-12 or abs(notional) < 1e-12 or T <= 0:
        # return 0.15 # Default vol
        return 0.0 # Return 0.0 and let the converged flag handle it

    target_price = float(np.clip(note_value / (notional * coupon), 1e-9, 1.0 - 1e-9))
    
    if use_bachelier:
        solver_function = calculate_implied_volatility_small_root_bachelier
    else:
        solver_function = calculate_implied_volatility_small_root_gbm
        
    vol, converged = solver_function(S, K, T, target_price)

    return vol if converged else 0.0


def test_implied_volatility():
    """두 모델의 내재변동성 역산 알고리즘 검증용 테스트."""
    test_cases = [
        # S, K, T, vol, name
        (0.0049, 0.0055, 1.0, 0.001, "OTM, Bachelier"),
        (0.0060, 0.0055, 1.0, 0.001, "ITM, Bachelier"),
        (0.0049, 0.0055, 1.0, 0.15, "OTM, GBM"),
        (0.0060, 0.0055, 1.0, 0.15, "ITM, GBM"),
        (0.0055, 0.00551, 1.0, 0.20, "ATM, Bachelier"),
    ]

    for idx, (S, K, T, vol, name) in enumerate(test_cases, 1):
        is_bachelier = "Bachelier" in name
        model_name = "Bachelier" if is_bachelier else "GBM"

        print("\n" + "=" * 70)
        print(f"테스트 {idx}: {name} ({model_name} 모델)")
        print(f"  입력: S={S*1e4:.1f}bp, K={K*1e4:.1f}bp, T={T}, 실제 변동성={vol}")
        
        if is_bachelier:
            known_price = bachelier_model_instance.price_digital(S, K, T, vol) # Use model instance
        else:
            known_price = gbm_model_instance.price_digital(S, K, T, vol) # Use model instance

        pricing_result = {
            'product': {
                'spread_asset_price': S, 'strike': K, 'expiry': T,
                'notional': 1.0, 'coupon': 1.0
            },
            'price': known_price
        }

        implied_vol = calculate_implied_volatility(pricing_result, use_bachelier=is_bachelier)
        error = abs(implied_vol - vol)

        print(f"  계산된 가격    = {known_price:.6f}")
        print(f"  역산된 변동성   = {implied_vol:.6f}")
        print(f"  오차            = {error:.2e}")
        print(f"  {'✓ PASS' if error < 1e-6 else '✗ FAIL'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_implied_volatility()