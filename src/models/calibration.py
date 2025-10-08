# cms_pricing/src/models/calibration.py

import sys
import os
import json
import time
from functools import lru_cache
from math import exp, log, sqrt
 
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, root_scalar
from scipy.stats import norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# 프로젝트 설정 파일에서 필요한 값들을 가져옵니다.
from cms_pricing.config import settings

# --- 모듈 상수 ---
SWAP_PAYMENT_FREQUENCY = 1.0 / settings.FREQ  # settings.py의 FREQ 사용

# =============================================================================
# 섹션 1: 시장 데이터 변환 (Black 모델 기반)
# =============================================================================

def calculate_forward_swap_rate_market(expiry: float, tenor: int, p_market_func: callable) -> tuple[float, float]:
    """주어진 할인 곡선(P)으로부터 선도 스왑 금리와 연금을 계산합니다."""
    payment_times = np.arange(expiry + SWAP_PAYMENT_FREQUENCY, expiry + tenor + 1e-8, SWAP_PAYMENT_FREQUENCY)
    
    if len(payment_times) == 0:
        return 0.0, 0.0
        
    annuity = sum(SWAP_PAYMENT_FREQUENCY * p_market_func(t) for t in payment_times)
    floating_pv = p_market_func(expiry) - p_market_func(expiry + tenor)
    
    if annuity < 1e-9:
        return 0.0, 0.0
        
    forward_rate = floating_pv / annuity
    return forward_rate, annuity

def black_swaption_price(forward_swap_rate: float, strike: float, volatility: float, expiry: float, annuity: float) -> float:
    """Black-76 모델을 사용하여 스왑션 가격을 계산합니다."""
    if volatility <= 1e-9 or expiry <= 1e-9:
        return max(0.0, (forward_swap_rate - strike)) * annuity
        
    d1 = (log(forward_swap_rate / strike) + 0.5 * volatility**2 * expiry) / (volatility * sqrt(expiry))
    d2 = d1 - volatility * sqrt(expiry)
    
    price = annuity * (forward_swap_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    return price

def build_market_prices_from_vol_surface(p_market_func: callable, surface_pct: list, expiry_labels: list, tenors: list) -> dict:
    """변동성 표면을 Black 모델을 이용해 시장 가격 표면으로 변환합니다."""
    label_to_year = {label: to_years(label) for label in expiry_labels}
    expiries_years = [label_to_year[lbl] for lbl in expiry_labels]
    
    market_prices = {}
    for i, T in enumerate(expiries_years):
        for j, tenor in enumerate(tenors):
            atm_vol = surface_pct[i][j] / 100.0
            forward_rate, annuity = calculate_forward_swap_rate_market(T, tenor, p_market_func)
            
            if forward_rate > 0 and annuity > 0:
                market_price = black_swaption_price(
                    forward_swap_rate=forward_rate, strike=forward_rate, # ATM 옵션
                    volatility=atm_vol, expiry=T, annuity=annuity
                )
                market_prices[(T, tenor)] = market_price
            else:
                market_prices[(T, tenor)] = 0.0
    return market_prices

# =============================================================================
# 섹션 2: G2++ 모델 가격 결정 함수
# =============================================================================

@lru_cache(maxsize=4096)
def _B(z, t, T):
    """G2++ 모델의 B(t,T) 보조 함수"""
    if abs(z) < 1e-9: return T - t
    return (1 - exp(-z * (T - t))) / z

@lru_cache(maxsize=4096)
def _V(t, T, a, b, sigma, eta, rho):
    """G2++ 모델의 V(t,T) 분산 함수"""
    T_m_t = T - t
    term1 = (sigma**2 / a**2) * (T_m_t + (2/a)*exp(-a*T_m_t) - (1/(2*a))*exp(-2*a*T_m_t) - 3/(2*a))
    term2 = (eta**2 / b**2) * (T_m_t + (2/b)*exp(-b*T_m_t) - (1/(2*b))*exp(-2*b*T_m_t) - 3/(2*b))
    term3 = (2*rho*sigma*eta/(a*b)) * (T_m_t + (exp(-a*T_m_t)-1)/a + (exp(-b*T_m_t)-1)/b - (exp(-(a+b)*T_m_t)-1)/(a+b))
    return term1 + term2 + term3

def price_european_swaption_g2_fast(
    params: dict, 
    p_market_func: callable, 
    expiry: float, 
    tenor: int,
    strike: float
    ) -> float:
    """Gauss-Hermite 구적법을 사용한 고속 G2++ 스왑션 가격 결정"""
    a, b, sigma, eta, rho = params['a'], params['b'], params['sigma'], params['eta'], params['rho']
    
    # 파라미터 유효성 검사
    if any(p <= 1e-9 for p in [a, b, sigma, eta]) or not (-1 < rho < 1): return 1e9

    try:
        sigma_x = sigma * sqrt((1 - exp(-2 * a * expiry)) / (2 * a))
        sigma_y = eta * sqrt((1 - exp(-2 * b * expiry)) / (2 * b))
        if sigma_x < 1e-9 or sigma_y < 1e-9: return 1e9
        rho_xy = (rho * sigma * eta / (a + b)) * (1 - exp(-(a + b) * expiry)) / (sigma_x * sigma_y)
    except (OverflowError, ValueError, ZeroDivisionError):
        return 1e9

    payment_times = np.arange(expiry + SWAP_PAYMENT_FREQUENCY, expiry + tenor + 1e-8, SWAP_PAYMENT_FREQUENCY)
    x_nodes, weights = np.polynomial.hermite.hermgauss(20)
    x_scaled = sigma_x * sqrt(2) * x_nodes
    
    integral_sum = 0.0
    for x, w in zip(x_scaled, weights):
        try:
            # y_bar(x)를 찾는 방정식
            def y_bar_eq(y):
                swap_pv = sum(
                    ( (1 + strike * SWAP_PAYMENT_FREQUENCY) if t == payment_times[-1] else strike * SWAP_PAYMENT_FREQUENCY) *
                    (p_market_func(t) / p_market_func(expiry)) *
                    exp(0.5 * (_V(expiry, t, a, b, sigma, eta, rho) - _V(0, t, a, b, sigma, eta, rho) + _V(0, expiry, a, b, sigma, eta, rho))
                        - _B(a, expiry, t) * x - _B(b, expiry, t) * y)
                    for t in payment_times
                )
                return swap_pv - 1.0
            
            y_bar = root_scalar(y_bar_eq, bracket=[-1.0, 1.0], method='brentq').root
            
            h1 = (y_bar / (sigma_y * sqrt(1 - rho_xy**2))) - (rho_xy * x) / (sigma_x * sqrt(1 - rho_xy**2))
            payoff_sum_term = sum(
                ((1 + strike * SWAP_PAYMENT_FREQUENCY) if t == payment_times[-1] else strike * SWAP_PAYMENT_FREQUENCY) *
                (p_market_func(t) / p_market_func(expiry)) *
                exp(0.5 * (_V(expiry, t, a, b, sigma, eta, rho) - _V(0, t, a, b, sigma, eta, rho) + _V(0, expiry, a, b, sigma, eta, rho))
                    - _B(a, expiry, t) * x - _B(b, expiry, t) * ( -0.5 * (1 - rho_xy**2) * sigma_y**2 * _B(b, expiry, t) + (rho_xy * sigma_y * x) / sigma_x)) *
                norm.cdf(-h1 - _B(b, expiry, t) * sigma_y * sqrt(1 - rho_xy**2))
                for t in payment_times
            )
            integral_sum += w * (norm.cdf(-h1) - payoff_sum_term)

        except (ValueError, OverflowError):
            continue
            
    return p_market_func(expiry) * integral_sum / sqrt(np.pi)


# =============================================================================
# 섹션 3: 병렬 보정 실행 함수
# =============================================================================

def _compute_single_error(details: tuple, params: dict, p_market_func: callable, market_prices: dict) -> float:
    """단일 스왑션의 제곱 오차를 계산 (병렬 처리용)"""
    expiry, tenor = details
    market_price = market_prices[details]
    forward_rate, _ = calculate_forward_swap_rate_market(expiry, tenor, p_market_func)
    
    model_price = price_european_swaption_g2_fast(params, p_market_func, expiry, tenor, forward_rate)
    
    return (model_price - market_price)**2

def _total_error_function(param_array: np.ndarray, p_market_func: callable, market_prices: dict, iteration: list) -> float:
    """최적화를 위한 전체 오차 함수 (병렬 처리)"""
    params = {'a': param_array[0], 'b': param_array[1], 'sigma': param_array[2], 'eta': param_array[3], 'rho': param_array[4]}
    iteration[0] += 1
    
    # joblib을 사용한 병렬 계산
    errors = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_single_error)(details, params, p_market_func, market_prices)
        for details in market_prices.keys()
    )
    
    total_error = sum(errors)
    
    print(f"\r  Iter: {iteration[0]:3d} | Error: {total_error:.8f} | a={params['a']:.3f}, b={params['b']:.3f}, "
          f"σ={params['sigma']:.4f}, η={params['eta']:.4f}, ρ={params['rho']:.3f}", end="")

    return total_error

# =============================================================================
# 섹션 4: 공개 API 함수
# =============================================================================

def calibrate_g2pp(p_market_func: callable, surface_pct: list, expiry_labels: list, tenors: list,
                   initial_params: dict = None) -> dict:
    """
    주어진 시장 데이터에 G2++ 모델 파라미터를 보정합니다.

    Args:
        p_market_func (callable): 할인 채권 가격 함수 P(t).
        surface_pct (list): % 단위의 스왑션 변동성 표면.
        expiry_labels (list): 만기 라벨 리스트 (예: ['1M', '1Y']).
        tenors (list): 테너 리스트 (예: [1, 5, 10]).
        initial_params (dict, optional): 최적화를 위한 초기 파라미터. Defaults to settings.

    Returns:
        dict: 보정된 파라미터 및 결과.
    """
    if initial_params is None:
        initial_params = settings.INITIAL_G2_PARAMS

    # 1. 변동성 표면을 시장 가격으로 변환
    print("  - 시장 가격 계산 중...")
    market_prices = build_market_prices_from_vol_surface(p_market_func, surface_pct, expiry_labels, tenors)
    
    # 2. 최적화 실행
    print("  - G2++ 파라미터 최적화 시작...")
    initial_array = np.array([initial_params[k] for k in ['a', 'b', 'sigma', 'eta', 'rho']])
    iteration_counter = [0] # 가변 객체를 이용한 카운터

    start_time = time.time()
    result = minimize(
        fun=_total_error_function,
        x0=initial_array,
        args=(p_market_func, market_prices, iteration_counter),
        method='Nelder-Mead',
        options={'maxiter': 50, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    elapsed = time.time() - start_time
    print("\n  - 최적화 완료.")

    # 3. 결과 정리 및 반환
    optimized_params = result.x
    final_params = {
        'a': optimized_params[0], 'b': optimized_params[1], 'sigma': optimized_params[2],
        'eta': optimized_params[3], 'rho': optimized_params[4],
        'final_error': result.fun,
        'iterations': result.nit,
        'elapsed_time_sec': elapsed,
        'status': result.message
    }

    # 보정된 파라미터를 파일에 저장
    save_calibrated_params(final_params)
    
    return final_params

def save_calibrated_params(params: dict, filename: str = None):
    """보정된 파라미터를 JSON 파일로 저장합니다."""
    if filename is None:
        filename = settings.CALIBRATED_PARAMS_FILE
    
    filepath = os.path.join(settings.DATA_DIR, filename)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    print(f"✓ 보정된 파라미터 저장 완료: {filepath}")

def load_calibrated_params(filename: str = None) -> dict:
    """저장된 파라미터를 JSON 파일에서 불러옵니다."""
    if filename is None:
        filename = settings.CALIBRATED_PARAMS_FILE
    
    filepath = os.path.join(settings.DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params

# --- 내부 유틸리티 함수 ---
def to_years(label: str) -> float:
    label = label.strip().upper()
    if label.endswith("M"): return int(label[:-1]) / 12.0
    if label.endswith("Y"): return float(label[:-1])
    raise ValueError(f"Unknown label format: {label}")