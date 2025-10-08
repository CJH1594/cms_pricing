#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_step_up_analysis.py (단위 변경 버전)
=======================================

Bachelier 모델을 사용하여 스텝업 노트의 예상 쿠폰과 리스크 지표(델타, 감마)를
금리 단위(%)로 계산하고, 헤지 효과를 분석합니다.
"""

import os
import sys
import numpy as np

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 모듈 임포트
from cms_pricing.src.models.analytic import BachelierModel
from cms_pricing.src.pricing.utils import load_pricing_results

class StepUpPricer:
    """Bachelier 모델 기반 스텝업 노트의 예상 쿠폰 및 리스크 계산기."""
    
    def __init__(self, product_spec: dict, S0: float, sigma: float):
        """
        Args:
            product_spec (dict): 'strikes', 'step_coupons', 'expiry' 포함
            S0 (float): 현재 시점의 기초자산(스프레드) 선도 가격
            sigma (float): 기초자산의 내재 변동성 (Bachelier vol)
        """
        self.strikes = product_spec['strikes']
        self.step_coupons = product_spec['step_coupons']
        self.T = product_spec['expiry']
        self.S0 = S0
        self.sigma = sigma
        self.model = BachelierModel()

    def _calculate_metrics(self, S: float) -> tuple:
        """주어진 S에 대해 예상 쿠폰, 델타, 감마를 계산."""
        expected_coupon = 0.0
        delta = 0.0
        gamma = 0.0

        for K, C_step in zip(self.strikes, self.step_coupons):
            # 개별 디지털 옵션의 가격 및 리스크를 계산하여 누적
            # price_digital은 확률 N(d)를 반환하므로, 쿠폰을 곱하면 기댓값이 됨
            expected_coupon += C_step * self.model.price_digital(S, K, self.T, self.sigma)
            delta += C_step * self.model.delta_digital(S, K, self.T, self.sigma)
            gamma += C_step * self.model.gamma_digital(S, K, self.T, self.sigma)
            
        return expected_coupon, delta, gamma

    def get_full_results(self, S_override: float = None) -> dict:
        """
        최종 예상 쿠폰, 델타, 감마를 계산하여 딕셔너리로 반환.
        S_override를 제공하면 해당 S값으로 재계산.
        """
        S = S_override if S_override is not None else self.S0
        
        expected_coupon, delta, gamma = self._calculate_metrics(S)
        
        return {
            "expected_coupon": expected_coupon,
            "delta": delta,
            "gamma": gamma,
        }


def main():
    print("=" * 70)
    print("Bachelier 모델 기반 스텝업 노트 리스크 분석 (금리 단위)")
    print("=" * 70)

    # --- 1. 입력 파라미터 정의 ---
    try:
        results = load_pricing_results()
        S0 = results['product']['spread_asset_price']
        sigma = results['stats']['spread_std'] / np.sqrt(results['product']['expiry'])
        print("✓ pricing_results.json에서 S0, sigma 값 로드 완료.")
    except (FileNotFoundError, KeyError):
        print("⚠ pricing_results.json 파일을 찾을 수 없어 기본값을 사용합니다.")
        S0 = 0.004877  # 현재 선도 스프레드
        sigma = 0.004728 # 내재 변동성
    
    product_spec = {
        'expiry': 1.0,
        'strikes': [0.0040, 0.0080, 0.0120],  # 40bp, 80bp, 120bp
        'step_coupons': [0.03, 0.03, 0.02],   # 각 레벨 통과 시 추가되는 쿠폰
    }

    print("\n[입력 파라미터]")
    print(f"  기초자산 (S0): {S0:.6f} ({S0*1e4:.2f} bp)")
    print(f"  내재 변동성 (σ): {sigma:.6f} ({sigma*1e4:.2f} bp)")
    print(f"  상품 구조 (최대 쿠폰 {sum(product_spec['step_coupons'])*100:.0f}%):")
    for i, (k, c) in enumerate(zip(product_spec['strikes'], product_spec['step_coupons'])):
        print(f"    - Level {i+1}: Spread > {k*1e4:.0f}bp 시 +{c*100:.0f}%")


    # --- 2. 예상 쿠폰 및 리스크 지표 계산 ---
    pricer = StepUpPricer(product_spec, S0, sigma)
    initial_results = pricer.get_full_results()
    
    initial_coupon = initial_results['expected_coupon']
    initial_delta = initial_results['delta']
    initial_gamma = initial_results['gamma']

    print("\n[초기 계산 결과]")
    print(f"  예상 쿠폰 (가격): {initial_coupon*100:.4f}%")
    print(f"  델타: {initial_delta:.2f} (bp당 쿠폰 변화율: {initial_delta/10000:.4f})")
    print(f"  감마: {initial_gamma:,.0f}")


    # --- 3. 헤지 시뮬레이션 ---
    print("\n[헤지 시뮬레이션]")
    shock = 0.0001  # +1bp 충격
    S_new = S0 + shock

    # 충격 후 실제 예상 쿠폰
    shock_results = pricer.get_full_results(S_override=S_new)
    true_new_coupon = shock_results['expected_coupon']
    
    # 1. 헤지 안했을 때 쿠폰 변화 (Unhedged P&L)
    coupon_change_actual = (true_new_coupon - initial_coupon) * 10000

    # 2. 델타 헤지만 했을 때 남는 손익 (Delta-Hedged P&L)
    coupon_change_delta_hedged = ((true_new_coupon - initial_coupon) - (initial_delta * shock)) * 10000

    # 3. 감마로 설명되는 손익 (P&L explained by Gamma)
    coupon_change_explained_by_gamma = (0.5 * initial_gamma * shock**2) * 10000
    
    print(f"  스프레드 +{shock*1e4:.0f}bp 상승 시 (결과는 bp 단위):")
    print(f"  - 실제 쿠폰 변화: {coupon_change_actual:+.4f} bp")
    print(f"  - 델타 헤지 후 잔여 변화: {coupon_change_delta_hedged:+.4f} bp")
    print(f"  - 감마로 설명되는 변화량: {coupon_change_explained_by_gamma:+.4f} bp")
    
    if abs(coupon_change_delta_hedged - coupon_change_explained_by_gamma) < 1e-5:
        print("  ✓ 델타 헤지 후 남은 변화량이 1/2 * Γ * (ΔS)² 공식으로 잘 설명됩니다.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()