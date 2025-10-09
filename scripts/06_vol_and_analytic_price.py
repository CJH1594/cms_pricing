#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_compare_mc_and_analytic.py
==============================

이 스크립트는 03번 스크립트에서 저장한 G2++ 몬테카를로 시뮬레이션 결과를
불러와 다음 두 가지 작업을 수행합니다:
1. 저장된 스프레드의 표준편차를 이용해 내재 바셸리에 변동성을 계산합니다.
2. 계산된 변동성과 바셸리에 분석 공식을 이용해 이론가를 계산하고,
   G2++ 몬테카를로 가격과 비교합니다.
"""

import os
import sys
import numpy as np
import json # Added for json.JSONDecodeError

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 모듈 임포트
from cms_pricing.src.models.analytic import BachelierModel
from cms_pricing.src.pricing.utils import load_pricing_results

def main() -> None:
    # --- 1. 저장된 가격 계산 결과 로드 ---
    print("--- 1. pricing_results.json 파일에서 데이터 로드 ---")
    try:
        results = load_pricing_results()
        print("✓ 데이터 로드 완료.")
    except (FileNotFoundError, json.JSONDecodeError) as e: # Add json.JSONDecodeError
        raise RuntimeError("가격 계산 결과 파일이 없습니다. 03_price_product.py를 먼저 실행하세요.") from e

    # 필요한 데이터 추출
    product = results['product']
    stats = results['stats']
    price_mc = results['price']
    
    T = product['expiry']
    K = product['strike']
    S0_spread = product['spread_asset_price']
    
    print(f"G2++ 몬테카를로 가격 (P_mc): {price_mc:.6f}")
    
    # --- 2. 저장된 통계치로부터 내재 변동성 계산 ---
    print("\n--- 2. 저장된 통계치로부터 내재 변동성 계산 ---")

    # StdDev(S_T) = σ * sqrt(T) 관계를 이용
    std_dev_of_spreads = stats['spread_std']
    implied_bachelier_vol = std_dev_of_spreads / np.sqrt(T) if T > 0 else 0.0

    print(f"만기 시점 스프레드 표준편차 (json에서 로드): {std_dev_of_spreads*1e4:.2f} bp")
    print(f"만기 (T): {T} 년")
    print(f"이를 이용해 계산한 바셸리에 변동성 (σ_implied): {implied_bachelier_vol:.6f}")

    # --- 3. 바셸리에 모델로 분석적 이론가 계산 ---
    print("\n--- 3. 바셸리에 분석 모델로 이론가 계산 ---")
    
    bachelier_model = BachelierModel()
    price_analytic = bachelier_model.price_digital(S0_spread, K, T, implied_bachelier_vol)
    
    print(f"현재 선도 스프레드 (S0): {S0_spread:.6f}")
    print(f"사용된 변동성 (σ_implied): {implied_bachelier_vol:.6f}")
    print(f"바셸리에 분석 모델 가격 (P_analytic): {price_analytic:.6f}")

    # --- 4. 결과 비교 ---
    print("\n--- 4. 최종 결과 비교 ---")
    price_diff = price_analytic - price_mc
    price_diff_pct = (price_diff / price_mc) * 100 if abs(price_mc) > 1e-9 else 0
    
    print(f"G2++ 몬테카를로 가격: {price_mc:.6f}")
    print(f"바셸리에 분석 모델 가격: {price_analytic:.6f}")
    print(f"가격 차이: {price_diff:+.6f} ({price_diff_pct:+.2f}%)")

    if abs(price_diff_pct) < 2.0:
        print("\n결론: 두 모델의 가격이 매우 유사하여, G2++ 모델의 동학이 바셸리에 모델로 잘 근사됨을 확인했습니다. 👍")
    else:
        print("\n결론: 두 모델의 가격에 상당한 차이가 존재합니다. 모델 간 근사의 한계를 확인했습니다.")


if __name__ == "__main__":
    main()