# -*- coding: utf-8 -*-
"""
G2++ 파라미터 보정 모듈
=======================

본 모듈은 시장의 ATM 스왑션 변동성 표면을 이용하여 G2++ 모델의 파라미터를
최적화합니다. 불필요한 중복 계산을 줄이고 성능을 개선하기 위해 캐싱과 병렬
계산을 적용합니다. 최적화 결과는 JSON 파일로 저장되며, 이후 실행에서는
바로 로드하여 시간을 절약할 수 있습니다.
"""

import json
import os
from functools import lru_cache
from math import log, sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize, root_scalar
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

# 패키지 표준 실행 방식에 맞는 절대 경로 임포트
from cms_pricing.config.settings import (
    DATA_DIR,
    CALIBRATED_PARAMS_FILE,
    SWAPTION_VOL_SURFACE,
    EXPIRY_LABELS,
    TENORS,
    INITIAL_G2_PARAMS,
)

# --- Black 모델 스왑션 가격 계산 ---

def black_swaption_price(forward_swap_rate: float,
                         strike: float,
                         volatility: float,
                         expiry: float,
                         annuity: float,
                         is_payer: bool = True) -> float:
    """Black 모델을 이용한 스왑션 가격을 계산합니다.

    변동성이나 만기가 0 이하이면 0을 반환합니다.
    """
    if volatility <= 1e-9 or expiry <= 1e-9:
        return 0.0
    
    # 등가격 옵션에서 log(1)=0이 되는 것을 피하기 위해 작은 epsilon 추가
    if abs(forward_swap_rate - strike) < 1e-9:
        d1 = (0.5 * volatility ** 2 * expiry) / (volatility * sqrt(expiry))
    else:
        d1 = (log(forward_swap_rate / strike) + 0.5 * volatility ** 2 * expiry) / (volatility * sqrt(expiry))
    
    d2 = d1 - volatility * sqrt(expiry)
    
    if is_payer: # Payer 스왑션 (금리 상승에 베팅)
        price = annuity * (forward_swap_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    else: # Receiver 스왑션 (금리 하락에 베팅)
        price = annuity * (strike * norm.cdf(-d2) - forward_swap_rate * norm.cdf(-d1))
    return price

# --- 시장 데이터 처리 ---

SWAP_PAYMENT_FREQUENCY = 0.5  # 고정금리 지급 주기 (반기)

def calculate_forward_swap_rate_market(expiry: float,
                                       tenor: float,
                                       P_market: callable) -> Tuple[float, float]:
    """t=0 시점의 시장 할인계수를 이용해 선도 스왑 금리와 애뉴어티(annuity)를 계산합니다."""
    # 지급 시점 계산 (만기 이후부터)
    payment_times = np.arange(expiry + SWAP_PAYMENT_FREQUENCY,
                              expiry + tenor + 1e-9,  # 끝점을 포함하기 위해 epsilon 추가
                              SWAP_PAYMENT_FREQUENCY)
    
    # 애뉴어티: 고정금리 1단위에 대한 현재가치 합
    annuity = sum(SWAP_PAYMENT_FREQUENCY * P_market(ti) for ti in payment_times)
    
    # 변동금리 현재가치
    floating_pv = P_market(expiry) - P_market(expiry + tenor)
    
    if annuity <= 1e-9:
        return 0.0, 0.0
        
    forward_rate = floating_pv / annuity
    return forward_rate, annuity


def build_market_prices_from_vol_surface(surface_pct: Iterable[Iterable[float]],
                                         expiry_labels: List[str],
                                         tenors: List[int],
                                         P_market: callable) -> Dict[Tuple[float, int], float]:
    """변동성 표면과 할인 곡선을 사용하여 시장 스왑션 가격을 구축합니다.

    Args:
        surface_pct: % 단위의 ATM 변동성을 담은 2차원 배열.
    """
    label_to_year = {"1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "10Y": 10.0}
    expiries_years = [label_to_year[lbl] for lbl in expiry_labels]
    
    market_prices: Dict[Tuple[float, int], float] = {}
    
    for i, T in enumerate(expiries_years):
        for j, tenor in enumerate(tenors):
            atm_vol = surface_pct[i][j] / 100.0
            forward_rate, annuity = calculate_forward_swap_rate_market(T, tenor, P_market)
            
            if forward_rate > 0.0 and annuity > 0.0:
                # ATM 스왑션이므로 행사가(strike)는 선도 스왑 금리와 동일
                price = black_swaption_price(forward_rate, forward_rate,
                                             atm_vol, T, annuity, is_payer=True)
                market_prices[(T, tenor)] = price
            else:
                market_prices[(T, tenor)] = 0.0
                
    return market_prices

# --- 캐싱을 적용한 G2++ 모델 구성 요소 ---

@lru_cache(maxsize=4096)
def B_cached(z: float, t: float, T: float) -> float:
    """캐싱된 G2++ 모델의 B(z, t, T) 함수."""
    if abs(z) < 1e-9:
        return T - t
    return (1.0 - np.exp(-z * (T - t))) / z


@lru_cache(maxsize=4096)
def calculate_V_cached(t: float, T: float, a: float, b: float,
                       sigma: float, eta: float, rho: float) -> float:
    """캐싱된 G2++ 모델의 V(t,T) 함수."""
    T_m_t = T - t
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return 0.0
        
    term1 = (sigma**2 / a**2) * (T_m_t + (2 / a) * np.exp(-a * T_m_t) - (1 / (2 * a)) * np.exp(-2 * a * T_m_t) - 3 / (2 * a))
    term2 = (eta**2 / b**2) * (T_m_t + (2 / b) * np.exp(-b * T_m_t) - (1 / (2 * b)) * np.exp(-2 * b * T_m_t) - 3 / (2 * b))
    term3 = (2 * rho * sigma * eta / (a * b)) * (T_m_t + (np.exp(-a * T_m_t) - 1) / a + (np.exp(-b * T_m_t) - 1) / b - (np.exp(-(a + b) * T_m_t) - 1) / (a + b))
    
    return term1 + term2 + term3


def price_swaption_g2_analytic(params: Dict[str, float],
                               swaption_details: Tuple[float, int, float, float],
                               P_market: callable) -> float:
    """가우스-에르미트 구적법을 이용한 G2++ 스왑션 분석적 해법."""
    a, b, sigma, eta, rho = params['a'], params['b'], params['sigma'], params['eta'], params['rho']
    # 파라미터 유효성 검사, 유효하지 않으면 큰 페널티 값 반환
    if any(p <= 0 for p in [a, b, sigma, eta]) or not (-1 <= rho <= 1):
        return 1e10

    expiry, tenor, strike, notional = swaption_details
    
    try:
        sigma_x = sigma * sqrt((1 - np.exp(-2 * a * expiry)) / (2 * a)) if a > 0 else 0.0
        sigma_y = eta * sqrt((1 - np.exp(-2 * b * expiry)) / (2 * b)) if b > 0 else 0.0
    except (OverflowError, ValueError):
        return 1e10 # 수치적 불안정성에 대한 페널티
        
    if sigma_x <= 1e-9 or sigma_y <= 1e-9: return 1e10

    try:
        rho_xy = (rho * sigma * eta / (a + b)) * (1 - np.exp(-(a + b) * expiry)) / (sigma_x * sigma_y)
    except (OverflowError, ZeroDivisionError):
        return 1e10

    payment_times = np.arange(expiry + SWAP_PAYMENT_FREQUENCY, expiry + tenor + 1e-9, SWAP_PAYMENT_FREQUENCY)
    
    x_nodes, weights = hermgauss(20) # 가우스-에르미트 구적법에 20개 노드 사용
    x_scaled = sigma_x * sqrt(2) * x_nodes
    
    integral_sum = 0.0
    
    for x, w in zip(x_scaled, weights):
        try:
            # 스왑 가치가 1이 되는 y_bar를 찾는 방정식
            def y_bar_equation(y: float) -> float:
                swap_pv = 0.0
                for t_i in payment_times:
                    c_i = strike * SWAP_PAYMENT_FREQUENCY if t_i != payment_times[-1] else 1.0 + strike * SWAP_PAYMENT_FREQUENCY
                    V_T_ti = calculate_V_cached(expiry, t_i, a, b, sigma, eta, rho)
                    A_T_ti = (P_market(t_i) / P_market(expiry)) * np.exp(0.5 * (V_T_ti - calculate_V_cached(0, t_i, a, b, sigma, eta, rho) + calculate_V_cached(0, expiry, a, b, sigma, eta, rho)))
                    P_T_ti = A_T_ti * np.exp(-B_cached(a, expiry, t_i) * x - B_cached(b, expiry, t_i) * y)
                    swap_pv += c_i * P_T_ti
                return swap_pv - 1.0

            y_bar = root_scalar(y_bar_equation, bracket=[-1.0, 1.0], method='brentq').root
            
            h1 = y_bar / (sigma_y * sqrt(1 - rho_xy**2)) - (rho_xy * x) / (sigma_x * sqrt(1 - rho_xy**2))
            
            payoff = norm.cdf(-h1)
            sum_term = 0.0
            for t_i in payment_times:
                c_i = strike * SWAP_PAYMENT_FREQUENCY if t_i != payment_times[-1] else 1.0 + strike * SWAP_PAYMENT_FREQUENCY
                V_T_ti = calculate_V_cached(expiry, t_i, a, b, sigma, eta, rho)
                A_T_ti = (P_market(t_i) / P_market(expiry)) * np.exp(0.5 * (V_T_ti - calculate_V_cached(0, t_i, a, b, sigma, eta, rho) + calculate_V_cached(0, expiry, a, b, sigma, eta, rho)))
                B_b = B_cached(b, expiry, t_i)
                lambda_i = c_i * A_T_ti * np.exp(-B_cached(a, expiry, t_i) * x)
                kappa_i = -B_b * ( (rho_xy * sigma_y * x) / sigma_x - 0.5 * (1 - rho_xy**2) * sigma_y**2 * B_b )
                h2 = h1 + B_b * sigma_y * sqrt(1 - rho_xy**2)
                sum_term += lambda_i * np.exp(kappa_i) * norm.cdf(-h2)

            payoff -= sum_term
            integral_sum += w * payoff
        except Exception:
            # 적분 중 발생하는 수치 오류는 무시하고 다음 노드로 진행
            continue
            
    integral = integral_sum / sqrt(np.pi)
    return notional * P_market(expiry) * integral

# --- 보정 목적 함수 ---

def _compute_single_error(expiry: float, tenor: int, market_price: float,
                          params: Dict[str, float], P_market: callable) -> float:
    """단일 스왑션에 대한 제곱 오차를 계산합니다."""
    forward_rate, _ = calculate_forward_swap_rate_market(expiry, tenor, P_market)
    swaption_details = (expiry, tenor, forward_rate, 1.0) # 보정을 위해 액면가는 1로 고정
    
    model_price = price_swaption_g2_analytic(params, swaption_details, P_market)
    
    error = model_price - market_price
    return error * error


def total_error_function(params_array: np.ndarray,
                         market_prices: Dict[Tuple[float, int], float],
                         P_market: callable) -> float:
    """모든 스왑션에 대한 제곱 오차의 합계를 반환합니다.

    joblib을 사용하여 병렬 계산을 수행합니다.
    """
    params = dict(zip(['a', 'b', 'sigma', 'eta', 'rho'], params_array))
    
    errors = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_single_error)(exp, ten, price, params, P_market)
        for (exp, ten), price in market_prices.items()
    )
    
    return sum(errors)

# --- 메인 보정 루틴 ---

def calibrate_g2pp(P_market: callable,
                   surface_pct: Iterable[Iterable[float]] = None,
                   expiry_labels: List[str] = None,
                   tenors: List[int] = None,
                   initial_params: Dict[str, float] = None,
                   save_results: bool = True,
                   maxiter: int = 50) -> Dict[str, float]:
    """ATM 스왑션 표면을 이용해 G2++ 파라미터를 보정합니다."""
    # 인자가 제공되지 않으면 settings의 기본값 사용
    surface_pct = surface_pct or SWAPTION_VOL_SURFACE
    expiry_labels = expiry_labels or EXPIRY_LABELS
    tenors = tenors or TENORS
    initial_params = initial_params or INITIAL_G2_PARAMS

    print("변동성 표면으로부터 시장 가격을 구축합니다...")
    market_prices = build_market_prices_from_vol_surface(surface_pct, expiry_labels, tenors, P_market)
    
    x0 = np.array(list(initial_params.values()))

    print("최적화를 시작합니다... (수 분이 소요될 수 있습니다)")
    result = minimize(
        total_error_function,
        x0,
        args=(market_prices, P_market),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-7}
    )
    print(f"최적화가 {result.nit}번의 반복 후 종료되었습니다.")

    optimized_params_array = result.x
    calibrated_params = {
        'a': float(optimized_params_array[0]),
        'b': float(optimized_params_array[1]),
        'sigma': float(optimized_params_array[2]),
        'eta': float(optimized_params_array[3]),
        'rho': float(optimized_params_array[4]),
        'final_error': float(result.fun),
        'iterations': int(result.nit),
        'status': result.message
    }

    if save_results:
        save_calibrated_params(calibrated_params)
        print(f"보정된 파라미터가 {CALIBRATED_PARAMS_FILE} 파일에 저장되었습니다.")
        
    return calibrated_params

# --- 파라미터 파일 입출력 ---

def _get_params_filepath(filename: str = CALIBRATED_PARAMS_FILE) -> str:
    """파라미터 파일의 절대 경로를 계산합니다."""
    return os.path.join(DATA_DIR, filename)


def save_calibrated_params(params: Dict[str, any], filename: str = CALIBRATED_PARAMS_FILE) -> None:
    """보정된 G2++ 파라미터를 JSON 파일에 저장합니다."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _get_params_filepath(filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_calibrated_params(filename: str = CALIBRATED_PARAMS_FILE) -> Dict[str, float]:
    """저장된 G2++ 파라미터를 JSON 파일에서 불러옵니다."""
    path = _get_params_filepath(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"보정 파일을 찾을 수 없습니다: {path}. 먼저 calibrate_g2pp를 실행하세요.")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 가격 결정 함수에서 사용할 핵심 파라미터만 float 형태로 반환
    core_params = {k: float(v) for k, v in data.items() if k in ['a', 'b', 'sigma', 'eta', 'rho']}
    return core_params