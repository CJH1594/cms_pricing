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
from scipy.stats import norm

from ..models.g2pp import calculate_forward_swap_rate
from ..pricing.simulation import simulate_paths
from ...config.settings import (
    SIMULATION_NUM_STEPS_PER_YEAR,
)


# def _phi(x: float) -> float:
#     """표준 정규 분포의 확률 밀도 함수(PDF)를 계산합니다."""
#     return float(norm.pdf(x))

def calculate_digital_bond_price(S0: float, K: float, T0: float, sigma: float) -> float:
    """블랙-숄즈 모델 기반의 디지털 채권 (현금 또는 무/유 상환) 가격을 계산합니다.
    이 경우, S0는 기초자산 가격, K는 스트라이크, T0는 만기, sigma는 변동성, r은 무위험 이자율입니다.
    """
    if sigma < 1e-9 or T0 < 1e-9:
        return 1.0 if S0 >= K else 0.0 # Degenerate case for zero volatility or zero time

    d2 = (np.log(S0 / K) + (- 0.5 * sigma**2) * T0) / (sigma * np.sqrt(T0))
    price = norm.cdf(d2)
    return float(price)

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
    num_steps = max(int(T_expiry * SIMULATION_NUM_STEPS_PER_YEAR), 100) # Use SIMULATION_NUM_STEPS_PER_YEAR
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

def calculate_digital_bond_delta(S0: float, K: float, T0: float, sigma: float) -> float:
    """디지털 채권의 델타 (Delt-B)를 계산합니다.
    Ref: Theorem 4. Delt-B Formula
    """
    d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T0) / (sigma * np.sqrt(T0))
    delt_B = norm.pdf(d1) / (K * sigma * np.sqrt(T0)) # Use norm.pdf directly
    return float(delt_B)

def calculate_digital_bond_vega(S: float, K: float, T: float, sigma: float) -> float:
    """
    디지털 본드의 Vega 계산
    
    Vega = -sigma/d1 * φ(d2)
    
    Args:
        S: 현재 스프레드 자산 가격
        K: 행사가격
        T: 만기까지 시간
        sigma: 변동성
    
    Returns:
        Vega (변동성에 대한 가격 민감도)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_T = np.sqrt(T)
    
    # d1, d2 계산
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # φ(d2): 표준정규분포 PDF
    phi_d2 = norm.pdf(d2)
    
    # Vega = -σ/d1 * φ(d2)
    # d1이 0에 가까워지면 수치적으로 불안정해지므로 d1이 매우 작은 경우 0을 반환
    if abs(d1) < 1e-6:  # 1e-10에서 1e-6으로 변경하여 더 보수적으로 처리
        return 0.0
    
    vega = -(sigma / d1) * phi_d2
    
    return vega

def calculate_spread_note_delta(S_prime_0: float, S0: float, K: float, T0: float, sigma: float) -> Tuple[float, float]:
    """스프레드 노트의 델타 (Delta_N)를 계산합니다.
    스프레드 (S_prime_0 - S0)를 기초 자산으로 하는 단일 디지털 채권의 델타를 기반으로 합니다.
    Ref: Definition 4. Spread Note
    """
    spread_value = S_prime_0 - S0
    # 여기서 sigma는 스프레드 자산의 변동성으로 가정합니다.
    delta_component = calculate_digital_bond_delta(spread_value, K, T0, sigma)
    return delta_component, -delta_component