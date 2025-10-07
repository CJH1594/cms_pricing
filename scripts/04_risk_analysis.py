#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_risk_analysis.py
===================

현재 프로젝트에서 위험 분석 및 헷징 기능은 구현되지 않았습니다.
이 스크립트는 향후 확장 가능성을 위한 자리표시자 역할을 합니다.

향후에는 델타, 감마 등 그릭스 계산과 헷징 시뮬레이션을 추가할 수 있습니다.
"""


import os
import sys
import numpy as np
from scipy.optimize import brentq

# Assuming cms_digital is in the parent directory of src/pricing
# Adjust the import path as necessary based on your project structure
from cms_pricing.src.market import create_continuous_zero_curve
from cms_pricing.src.market import load_market_data # bootstrap_if_needed 대신 직접 로드
from cms_pricing.src.models import phi_g2pp_factory, calculate_V, calculate_forward_swap_rate
from cms_pricing.src.pricing.cms_digital import calculate_spread_note_delta, calculate_digital_bond_price
from cms_pricing.src.pricing import load_pricing_results

def _calculate_implied_volatility(
    spread_note_value: float,
    product: dict,
    spread_asset_price: float,
    K: float,
    T0: float
) -> float:
    """
    디지털 본드 가격을 이용해 내재 변동성 (Implied Volatility)을 역산합니다.
    """
    # 목표 디지털 본드 가격 계산
    # spread_note_value = notional * coupon * E[D_T * I(Spread > K)]
    # calculate_digital_bond_price는 만기 시 1단위 현금을 지급하는 디지털 옵션 가격
    # 따라서 목표 디지털 본드 가격은 spread_note_value / (notional * coupon)이 됩니다.
    # 단, coupon이 0에 가까울 경우를 대비하여 처리합니다.
    target_bond_price = spread_note_value / product['notional']
    if product['coupon'] > 1e-9:
        target_bond_price /= product['coupon']
    else:
        # 쿠폰이 0에 가깝다면, 가격 자체를 목표로 하거나 다른 처리가 필요
        print("경고: 상품 쿠폰이 매우 작습니다. 내재 변동성 역산에 오차가 있을 수 있습니다.")

    def target_function_for_iv(vol: float) -> float:
        # 블랙-숄즈 모델 기반 디지털 콜 옵션 가격을 사용하여 목표 가격과 비교
        bs_price = calculate_digital_bond_price(spread_asset_price, K, T0, vol)
        return bs_price - target_bond_price

    # 변동성 탐색 범위 (0.01% ~ 200%)
    low_vol = 1e-4
    high_vol = 2.0
    implied_sigma = 0.15 # 찾지 못하면 초기 임시 값을 사용

    try:
        # brentq는 f(a)와 f(b)의 부호가 다를 때만 작동
        # 따라서 경계 값에서의 함수 부호 확인이 중요합니다.
        if target_function_for_iv(low_vol) * target_function_for_iv(high_vol) < 0:
            implied_sigma = brentq(target_function_for_iv, low_vol, high_vol)
        else:
            print("내재 변동성을 찾기 위한 적절한 구간을 찾을 수 없습니다. 기본값을 사용합니다.")
            # 필요하다면 여기에서 더 넓은 범위 탐색 또는 다른 알고리즘 시도
    except ValueError as e:
        print(f"내재 변동성을 찾을 수 없습니다: {e}")
        print(f"  현재 가격 (Spread Note Value): {spread_note_value:.4f}")
        print(f"  목표 본드 가격: {target_bond_price:.4f}")
        print(f"  Low Vol에서의 가격 (BS): {calculate_digital_bond_price(spread_asset_price, K, T0, low_vol):.4f}")
        print(f"  High Vol에서의 가격 (BS): {calculate_digital_bond_price(spread_asset_price, K, T0, high_vol):.4f}")

    return implied_sigma


def _load_and_prepare_data():
    """
    가격 계산 결과를 로드하고, 시장 데이터를 부트스트랩하며, G2++ 모델 관련 함수들을 생성합니다.
    """
    try:
        pricing_results = load_pricing_results()
    except FileNotFoundError as e:
        raise RuntimeError("가격 계산 결과 파일이 없습니다. 먼저 03_price_product.py를 실행하세요.") from e

    product = pricing_results['product']
    spread_note_value = pricing_results['price']
    params = pricing_results['params']

    # 시장 데이터는 01_bootstrap_curve.py에서 미리 계산되어 저장되어야 함
    try:
        par_full, P_year = load_market_data()
    except FileNotFoundError as e:
        raise RuntimeError("시장 데이터 파일이 없습니다. 먼저 01_bootstrap_curve.py를 실행하세요.") from e
    
    zero_curve = create_continuous_zero_curve(P_year)
    P_market = lambda t: np.exp(-zero_curve(t) * t)
    V_0_func = lambda t: calculate_V(0.0, t, params)
    phi_func, f0_func = phi_g2pp_factory(P_year, params['a'], params['b'], params['sigma'], params['eta'], params['rho'])

    return product, spread_note_value, params, zero_curve, P_market, V_0_func, phi_func, f0_func


def _calculate_initial_greeks(
    product: dict,
    params: dict,
    zero_curve, P_market, V_0_func,
    spread_note_value: float
):
    """
    초기 상태에서 델타 및 포트폴리오 가치를 계산합니다.
    """
    T0 = product['expiry']
    K = product['strike']
    x_0 = 0.0
    y_0 = 0.0

    S_prime_0 = calculate_forward_swap_rate(
        t_future=T0,
        tenor=product['tenor_long'],
        x_t=x_0,
        y_t=y_0,
        params=params,
        P_market=P_market,
        V_0_func=V_0_func
    )
    S0 = calculate_forward_swap_rate(
        t_future=T0,
        tenor=product['tenor_short'],
        x_t=x_0,
        y_t=y_0,
        params=params,
        P_market=P_market,
        V_0_func=V_0_func
    )

    print("로드된 상품 스펙:")
    print(f"  만기 (T0): {T0}년")
    print(f"  CMS Long (S_prime_0): {S_prime_0:.4f}")
    print(f"  CMS Short (S0): {S0:.4f}")
    print(f"  Strike (K): {K*1e4:.0f} bp")
    print(f"  디지털 CMS 노트 가치: {spread_note_value:.2f}")

    spread_asset_price = S_prime_0 - S0

    sigma = _calculate_implied_volatility(
        spread_note_value,
        product,
        spread_asset_price,
        K,
        T0
    )
    print(f"  내재 변동성 (Implied Sigma): {sigma:.4f}")

    delta_N = calculate_spread_note_delta(S_prime_0, S0, K, T0, sigma)
    print(f"계산된 스프레드 노트 델타 (Delta_N): ({delta_N[0]:.4f}, {delta_N[1]:.4f})")

    portfolio_value = spread_note_value - delta_N[0] * S_prime_0 - delta_N[1] * S0
    print(f"초기 델타 헤지 포트폴리오 가치: {portfolio_value:.4f}")

    return S_prime_0, S0, K, T0, sigma, delta_N, portfolio_value


def _simulate_market_shock_and_rebalance(
    S_prime_0: float, S0: float, K: float, T0: float, sigma: float, delta_N: tuple, spread_note_value: float
):
    """
    시장 변화를 시뮬레이션하고 포트폴리오를 리밸런싱합니다.
    """
    shock = 0.01
    S_prime_new = S_prime_0 + shock
    S_new = S0 + shock

    delta_N_new = calculate_spread_note_delta(S_prime_new, S_new, K, T0, sigma)

    spread_note_value_new = spread_note_value + 0.1 * shock # 가상의 변화

    portfolio_value_new = spread_note_value_new - delta_N[0] * S_prime_new - delta_N[1] * S_new
    print(f"변화 후 델타 헤지 포트폴리오 가치 (초기 델타 사용): {portfolio_value_new:.4f}")

    rebalanced_portfolio_value = spread_note_value_new - delta_N_new[0] * S_prime_new - delta_N_new[1] * S_new
    print(f"변화 후 리밸런싱된 델타 헤지 포트폴리오 가치 (새로운 델타 사용): {rebalanced_portfolio_value:.4f}")


def main() -> None:
    print("스프레드 노트 델타 헤지 시뮬레이션")

    product, spread_note_value, params, zero_curve, P_market, V_0_func, phi_func, f0_func = _load_and_prepare_data()

    S_prime_0, S0, K, T0, sigma, delta_N, portfolio_value = _calculate_initial_greeks(
        product, params, zero_curve, P_market, V_0_func, spread_note_value
    )

    _simulate_market_shock_and_rebalance(
        S_prime_0, S0, K, T0, sigma, delta_N, spread_note_value
    )


if __name__ == "__main__":
    main()