#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_price_product.py
===================

이 스크립트는 보정된 G2++ 파라미터와 시장 할인계수를 이용하여
디지털 CMS 스프레드 노트의 가격을 계산합니다. 필요한 데이터가 없으면
사전에 01과 02 스크립트를 실행해야 합니다.
"""

import os
import sys
import numpy as np

# PYTHONPATH 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cms_pricing.src.market import bootstrap_if_needed, create_continuous_zero_curve
from cms_pricing.src.models import make_lnP_spline, phi_g2pp_factory, calculate_V, load_calibrated_params
from cms_pricing.src.pricing import price_digital_cms_spread


def main() -> None:
    # 1. 시장 데이터 로드
    par_full, P_year = bootstrap_if_needed()
    zero_curve = create_continuous_zero_curve(P_year)
    P_market = lambda t: np.exp(-zero_curve(t) * t)
    # 2. 파라미터 로드
    try:
        params = load_calibrated_params()
    except FileNotFoundError as e:
        raise RuntimeError("보정된 파라미터 파일이 없습니다. 먼저 02_calibrate_g2pp.py를 실행하세요.") from e
    # 3. φ(t)와 f(0,t) 함수 생성
    phi_func, f0_func = phi_g2pp_factory(P_year, params['a'], params['b'], params['sigma'], params['eta'], params['rho'])
    # V_0(t) 함수 생성
    V_0_func = lambda t: calculate_V(0.0, t, params)
    # 4. 상품 정의
    product = {
        'expiry': 1.0,
        'tenor_long': 10.0,
        'tenor_short': 2.0,
        'strike': 0.0055,
        'coupon': 0.07,
        'notional': 100.0,
    }
    print("상품 스펙:")
    print(f"  만기: {product['expiry']}년")
    print(f"  CMS: {product['tenor_long']:.0f}Y - {product['tenor_short']:.0f}Y")
    print(f"  Strike: {product['strike']*1e4:.0f} bp")
    print(f"  Coupon: {product['coupon']*100:.2f}%")
    print(f"  Notional: {product['notional']}")
    # 5. 가격 계산
    price, spreads, D_paths, stats = price_digital_cms_spread(
        params,
        (product['expiry'], product['tenor_long'], product['tenor_short'],
         product['strike'], product['coupon'], product['notional']),
        P_market,
        V_0_func,
        phi_func,
        num_paths=50000,
    )
    print("\n가격 결과:")
    print(f"  현재가치: {price:.2f}")
    print(f"  액면가 대비: {price / product['notional'] * 100:.2f}%")
    print(f"  지급 확률: {stats['payout_prob']:.2%}")
    print(f"  공정 쿠폰: {stats['fair_coupon']*100:.2f}%")
    # 통계 출력
    print("\n추가 통계:")
    for key in ['mean_D', 'df_T', 'p_T', 'discount_error_pct', 'spread_mean', 'spread_std']:
        val = stats[key]
        if 'spread' in key:
            print(f"  {key}: {val*1e4:.2f} bp")
        elif 'error' in key:
            print(f"  {key}: {val:.4f}%")
        else:
            print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    main()