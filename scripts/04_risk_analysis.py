#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_risk_analysis.py
===================

디지털 CMS 스프레드 노트의 위험 분석 및 델타 헤지 시뮬레이션 스크립트.
"""

import os
import sys
import numpy as np

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cms_pricing.src.pricing import load_pricing_results
from cms_pricing.src.pricing.cms_digital import calculate_spread_note_delta
from cms_pricing.src.volatility.implied_volatility import calculate_implied_volatility


def _load_product() -> dict:
    """가격 계산 결과에서 상품 정보를 읽어온다."""
    try:
        pricing_results = load_pricing_results()
    except FileNotFoundError as e:
        raise RuntimeError("가격 계산 결과 파일이 없습니다. 03_price_product.py를 먼저 실행하세요.") from e

    return pricing_results


def _calculate_initial_delta(pricing_results: dict) -> tuple:
    """상품 정보와 가격으로부터 내재변동성 및 델타를 계산한다."""
    prod = pricing_results["product"]

    # 내재변동성 역산 (뉴턴-랩슨)
    implied_vol = calculate_implied_volatility(pricing_results, use_bachelier= True)

    # 델타 계산 (Black-숄즈 디지털 옵션 델타 공식)
    delta_long, delta_short = calculate_spread_note_delta(
        prod["S_prime_0"],
        prod["S0"],
        prod["strike"],
        prod["expiry"],
        implied_vol,
    )

    # 현재 포트폴리오 가치 (노트 가치 - 델타 * 기초자산 가격)
    note_value = pricing_results["price"]
    portfolio_value = note_value - delta_long * prod["S_prime_0"] - delta_short * prod["S0"]

    print("로드된 상품 스펙:")
    print(f"  만기 (T0): {prod['expiry']}년")
    print(f"  CMS Long (S_prime_0): {prod['S_prime_0']:.6f}")
    print(f"  CMS Short (S0): {prod['S0']:.6f}")
    print(f"  Strike (K): {prod['strike']:.6f}")
    print(f"  디지털 CMS 노트 가치: {note_value:.6f}")
    print(f"  내재 변동성 (σ): {implied_vol:.6f}")
    print(f"계산된 스프레드 노트 델타: (Long leg Δ = {delta_long:.6f}, Short leg Δ = {delta_short:.6f})")
    print(f"초기 델타 헤지 포트폴리오 가치: {portfolio_value:.6f}")

    return (implied_vol, delta_long, delta_short, portfolio_value)


def _simulate_shock(
    prod: dict,
    implied_vol: float,
    delta_long: float,
    delta_short: float,
    note_value: float,
) -> None:
    """CMS 금리에 작은 평행 이동을 가정하고 델타 헤지 포트폴리오의 변화를 평가한다."""
    shock = 0.01  # 1bp = 0.0001, 0.01은 100bp(1%) 평행 이동 예시
    S_prime_new = prod["S_prime_0"] + shock
    S_new = prod["S0"] + shock

    # 새로운 델타 (델타 재계산; 여기서는 변동성 불변 가정)
    new_delta_long, new_delta_short = calculate_spread_note_delta(
        S_prime_new, S_new, prod["strike"], prod["expiry"], implied_vol
    )

    # 노트 가격도 평행 이동만큼 변화했다고 가정 (단순 예시)
    # 실제로는 가중치를 고려한 가격 변화를 시뮬레이션해야 한다.
    note_value_new = note_value + 0.1 * shock

    # 기존 델타를 사용한 포트폴리오 가치 변화
    portfolio_value_old_hedge = note_value_new - delta_long * S_prime_new - delta_short * S_new
    # 새로운 델타로 리밸런싱한 포트폴리오 가치
    portfolio_value_new_hedge = note_value_new - new_delta_long * S_prime_new - new_delta_short * S_new

    print(f"\n시장 금리 {shock:.2%} 평행 이동 후:")
    print(f"  기존 델타 사용 포트폴리오 가치: {portfolio_value_old_hedge:.6f}")
    print(f"  재계산한 델타로 리밸런싱한 가치: {portfolio_value_new_hedge:.6f}")


def main() -> None:
    # 상품 정보 및 가격 불러오기
    pricing_results = _load_product()
    prod = pricing_results["product"]

    # 초기 델타 및 포트폴리오 가치 계산
    implied_vol, delta_long, delta_short, portfolio_value = _calculate_initial_delta(pricing_results)

    # 시장 충격을 가정하여 포트폴리오 변화 시뮬레이션
    _simulate_shock(prod, implied_vol, delta_long, delta_short, pricing_results["price"])


if __name__ == "__main__":
    main()