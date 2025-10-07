# -*- coding: utf-8 -*-
"""
G2++ 파라미터 보정 모듈
=======================

본 모듈은 시장 ATM 스왑션 변동성 표면을 이용하여 G2++ 모델의 파라미터를
최적화합니다. 불필요한 중복을 줄이고 성능을 개선하기 위해 캐싱과 병렬
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
from scipy.optimize import minimize
from scipy.stats import norm

from ...config.settings import (
    DATA_DIR,
    CALIBRATED_PARAMS_FILE,
    EXPIRY_LABELS,
    TENORS,
    INITIAL_G2_PARAMS,
)


def black_swaption_price(forward_swap_rate: float,
                         strike: float,
                         volatility: float,
                         expiry: float,
                         annuity: float,
                         is_payer: bool = True) -> float:
    """Black 모델을 이용한 ATM 스왑션 가격 계산.

    volatility가 0이거나 expiry가 0이면 0을 반환합니다.
    """
    if volatility <= 0.0 or expiry <= 0.0:
        return 0.0
    d1 = (log(forward_swap_rate / strike) + 0.5 * volatility ** 2 * expiry) / (volatility * sqrt(expiry))
    d2 = d1 - volatility * sqrt(expiry)
    if is_payer:
        price = annuity * (forward_swap_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        price = annuity * (strike * norm.cdf(-d2) - forward_swap_rate * norm.cdf(-d1))
    return price


SWAP_PAYMENT_FREQUENCY = 0.5  # 고정 다리 지급 빈도 (semi-annual)


def calculate_forward_swap_rate_market(expiry: float,
                                       tenor: float,
                                       P_market: callable) -> Tuple[float, float]:
    """시장 할인계수를 이용해 t=0 시점의 선도 스왑 금리와 annuity를 계산합니다.

    만기 expiry 이후부터 expiry+tenor까지 0.5년 간격으로 지급되는 고정 다리를 가정합니다.
    """
    payment_times = np.arange(expiry + SWAP_PAYMENT_FREQUENCY,
                              expiry + tenor + 0.01,
                              SWAP_PAYMENT_FREQUENCY)
    annuity = sum(SWAP_PAYMENT_FREQUENCY * P_market(ti) for ti in payment_times)
    floating_pv = P_market(expiry) - P_market(expiry + tenor)
    if annuity == 0.0:
        return 0.0, 0.0
    forward_rate = floating_pv / annuity
    return forward_rate, annuity


def build_market_prices_from_vol_surface(surface_pct: Iterable[Iterable[float]],
                                         expiry_labels: List[str],
                                         tenors: List[int],
                                         P_market: callable) -> Dict[Tuple[float, int], float]:
    """변동성 표면과 할인계수 곡선을 사용하여 시장 스왑션 가격을 계산합니다.

    surface_pct는 만기×텐서 형태의 2차원 배열로, 각 값은 % 단위의 ATM 변동성을
    의미합니다. forward_rate와 strike를 동일하게 설정하여 ATM 스왑션 가격을
    구합니다.
    """
    # 라벨을 연 단위로 변환
    label_to_year = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}
    expiries_years = [label_to_year[lbl] for lbl in expiry_labels]
    market_prices: Dict[Tuple[float, int], float] = {}
    for i, T in enumerate(expiries_years):
        for j, tenor in enumerate(tenors):
            atm_vol = surface_pct[i][j] / 100.0
            forward_rate, annuity = calculate_forward_swap_rate_market(T, tenor, P_market)
            if forward_rate > 0.0 and annuity > 0.0:
                price = black_swaption_price(forward_rate, forward_rate,
                                             atm_vol, T, annuity, is_payer=True)
                market_prices[(T, tenor)] = price
            else:
                market_prices[(T, tenor)] = 0.0
    return market_prices


@lru_cache(maxsize=2048)
def B_cached(z: float, t: float, T: float) -> float:
    """캐싱된 B 함수"""
    if abs(z) < 1e-9:
        return T - t
    return (1.0 - np.exp(-z * (T - t))) / z


@lru_cache(maxsize=2048)
def calculate_V_cached(t: float, T: float, a: float, b: float,
                       sigma: float, eta: float, rho: float) -> float:
    """캐싱된 V(t,T) 함수"""
    T_m_t = T - t
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return 0.0
    term1 = (sigma ** 2 / a ** 2) * (T_m_t + (2 / a) * np.exp(-a * T_m_t) - (1 / (2 * a)) * np.exp(-2 * a * T_m_t) - 3 / (2 * a))
    term2 = (eta ** 2 / b ** 2) * (T_m_t + (2 / b) * np.exp(-b * T_m_t) - (1 / (2 * b)) * np.exp(-2 * b * T_m_t) - 3 / (2 * b))
    term3 = (2 * rho * sigma * eta / (a * b)) * (T_m_t + (np.exp(-a * T_m_t) - 1) / a + (np.exp(-b * T_m_t) - 1) / b - (np.exp(-(a + b) * T_m_t) - 1) / (a + b))
    return term1 + term2 + term3


def price_european_swaption_g2_fast(params_dict: Dict[str, float],
                                    swaption_details: Tuple[float, int, float, float],
                                    P_market: callable) -> float:
    """Gauss-Hermite 적분을 이용한 G2++ 스왑션 가격 계산.

    swaption_details = (expiry, tenor, strike, notional)
    """
    a = params_dict['a']; b = params_dict['b']; sigma = params_dict['sigma']; eta = params_dict['eta']; rho = params_dict['rho']
    if any(p <= 0 for p in [a, b, sigma, eta]) or not (-1 <= rho <= 1):
        return 1e10
    expiry, tenor, strike, notional = swaption_details
    # 사전 계산
    try:
        sigma_x = sigma * sqrt((1 - np.exp(-2 * a * expiry)) / (2 * a)) if a > 0 else 0.0
        sigma_y = eta * sqrt((1 - np.exp(-2 * b * expiry)) / (2 * b)) if b > 0 else 0.0
    except (OverflowError, ValueError):
        return 1e10
    if sigma_x <= 1e-9 or sigma_y <= 1e-9:
        return 1e10
    try:
        rho_xy = (rho * sigma * eta / (a + b)) * (1 - np.exp(-(a + b) * expiry)) / (sigma_x * sigma_y)
    except (OverflowError, ZeroDivisionError):
        return 1e10
    M_x_T = 0.0; M_y_T = 0.0
    payment_times = np.arange(expiry + SWAP_PAYMENT_FREQUENCY,
                              expiry + tenor + 0.01,
                              SWAP_PAYMENT_FREQUENCY)
    # Gauss-Hermite 노드 및 가중치(20점 사용)
    from numpy.polynomial.hermite import hermgauss
    x_nodes, weights = hermgauss(20)
    x_scaled = M_x_T + sigma_x * sqrt(2) * x_nodes
    integral_sum = 0.0
    for x, w in zip(x_scaled, weights):
        try:
            # y_bar 방정식 정의: 스왑 가치=1을 만족하는 y
            def y_bar_equation(y: float) -> float:
                swap_pv = 0.0
                for t_i in payment_times:
                    c_i = strike * SWAP_PAYMENT_FREQUENCY if t_i != payment_times[-1] else 1.0 + strike * SWAP_PAYMENT_FREQUENCY
                    V_T_ti = calculate_V_cached(expiry, t_i, a, b, sigma, eta, rho)
                    V_0_ti = calculate_V_cached(0.0, t_i, a, b, sigma, eta, rho)
                    V_0_T = calculate_V_cached(0.0, expiry, a, b, sigma, eta, rho)
                    try:
                        A_T_ti = (P_market(t_i) / P_market(expiry)) * np.exp(0.5 * (V_T_ti - V_0_ti + V_0_T))
                        P_T_ti = A_T_ti * np.exp(-B_cached(a, expiry, t_i) * x - B_cached(b, expiry, t_i) * y)
                    except (OverflowError, ValueError):
                        return 1.0  # 비정상적인 값 방지
                    swap_pv += c_i * P_T_ti
                return swap_pv - 1.0
            # Brent 방법으로 y_bar 찾기
            from scipy.optimize import root_scalar
            try:
                result = root_scalar(y_bar_equation, bracket=[-1.0, 1.0], method='brentq', xtol=1e-6)
            except ValueError:
                return 1e10
            y_bar = result.root
            h1 = (y_bar - M_y_T) / (sigma_y * sqrt(1 - rho_xy ** 2)) - (rho_xy * (x - M_x_T)) / (sigma_x * sqrt(1 - rho_xy ** 2))
            payoff = norm.cdf(-h1)
            sum_term = 0.0
            for t_i in payment_times:
                c_i = strike * SWAP_PAYMENT_FREQUENCY if t_i != payment_times[-1] else 1.0 + strike * SWAP_PAYMENT_FREQUENCY
                V_T_ti = calculate_V_cached(expiry, t_i, a, b, sigma, eta, rho)
                V_0_ti = calculate_V_cached(0.0, t_i, a, b, sigma, eta, rho)
                V_0_T = calculate_V_cached(0.0, expiry, a, b, sigma, eta, rho)
                A_T_ti = (P_market(t_i) / P_market(expiry)) * np.exp(0.5 * (V_T_ti - V_0_ti + V_0_T))
                B_b = B_cached(b, expiry, t_i)
                lambda_i = c_i * A_T_ti * np.exp(-B_cached(a, expiry, t_i) * x)
                kappa_i = -B_b * (M_y_T - 0.5 * (1 - rho_xy ** 2) * sigma_y ** 2 * B_b + (rho_xy * sigma_y * (x - M_x_T)) / sigma_x)
                h2 = h1 + B_b * sigma_y * sqrt(1 - rho_xy ** 2)
                sum_term += lambda_i * np.exp(kappa_i) * norm.cdf(-h2)
            payoff -= sum_term
            integral_sum += w * payoff
        except Exception:
            # 예기치 못한 오류는 무시하고 다음 노드로 진행
            continue
    integral = integral_sum / sqrt(np.pi)
    return notional * P_market(expiry) * integral


def compute_single_error(expiry: float,
                         tenor: int,
                         market_price: float,
                         params_dict: Dict[str, float],
                         P_market: callable) -> float:
    """개별 스왑션에 대한 제곱 오차를 계산합니다."""
    forward_rate, _ = calculate_forward_swap_rate_market(expiry, tenor, P_market)
    swaption_details = (expiry, tenor, forward_rate, 1.0)
    model_price = price_european_swaption_g2_fast(params_dict, swaption_details, P_market)
    error = model_price - market_price
    return error * error


def total_error_function_g2(g2_params_array: np.ndarray,
                            market_prices: Dict[Tuple[float, int], float],
                            P_market: callable) -> float:
    """전체 스왑션에 대한 제곱 오차 합계를 반환합니다.

    joblib을 사용하여 병렬 계산을 지원합니다.
    """
    params_dict = {
        'a': g2_params_array[0],
        'b': g2_params_array[1],
        'sigma': g2_params_array[2],
        'eta': g2_params_array[3],
        'rho': g2_params_array[4],
    }
    # 병렬로 각 스왑션 오차 계산
    errors = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_single_error)(exp, ten, price, params_dict, P_market)
        for (exp, ten), price in market_prices.items()
    )
    total_error = sum(errors)
    return total_error


def calibrate_g2pp(surface_pct: Iterable[Iterable[float]],
                   expiry_labels: List[str] = None,
                   tenors: List[int] = None,
                   P_market: callable = None,
                   initial_params: Dict[str, float] = None,
                   save_file: bool = True,
                   filename: str = CALIBRATED_PARAMS_FILE,
                   maxiter: int = 50) -> Dict[str, float]:
    """ATM 스왑션 표면을 이용해 G2++ 파라미터를 보정합니다.

    surface_pct는 [%] 단위의 2차원 배열입니다. expiry_labels와 tenors가
    제공되지 않으면 settings의 기본값을 사용합니다. 초기값은
    INITIAL_G2_PARAMS를 사용하며, 저장 옵션이 True이면 결과를 JSON으로
    저장합니다.
    """
    if expiry_labels is None:
        expiry_labels = EXPIRY_LABELS
    if tenors is None:
        tenors = TENORS
    if initial_params is None:
        initial_params = INITIAL_G2_PARAMS
    if P_market is None:
        raise ValueError("P_market 함수가 필요합니다.")
    market_prices = build_market_prices_from_vol_surface(surface_pct, expiry_labels, tenors, P_market)
    x0 = np.array([initial_params['a'], initial_params['b'], initial_params['sigma'], initial_params['eta'], initial_params['rho']])
    # Nelder-Mead 알고리즘을 사용한 최소화
    result = minimize(
        total_error_function_g2,
        x0,
        args=(market_prices, P_market),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    optimized_params = result.x
    params_dict = {
        'a': float(optimized_params[0]),
        'b': float(optimized_params[1]),
        'sigma': float(optimized_params[2]),
        'eta': float(optimized_params[3]),
        'rho': float(optimized_params[4]),
        'final_error': float(result.fun),
        'nit': int(result.nit),
    }
    if save_file:
        save_calibrated_params(params_dict, filename)
    return params_dict


def _get_calibrated_path(filename: str = CALIBRATED_PARAMS_FILE) -> str:
    """보정 파라미터 파일의 절대 경로를 계산"""
    return os.path.join(DATA_DIR, filename)


def save_calibrated_params(params: Dict[str, float], filename: str = CALIBRATED_PARAMS_FILE) -> None:
    """보정된 G2++ 파라미터를 JSON 파일에 저장합니다."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _get_calibrated_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_calibrated_params(force: bool = False,
                           filename: str = CALIBRATED_PARAMS_FILE) -> Dict[str, float]:
    """저장된 파라미터를 로드하거나, force=True이면 새로 보정합니다.

    force=True인 경우 기존 파일을 무시하고 새로 보정합니다. force=False이고
    파일이 존재하지 않으면 FileNotFoundError가 발생합니다.
    """
    path = _get_calibrated_path(filename)
    if not force and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items() if k in ['a', 'b', 'sigma', 'eta', 'rho']}
    else:
        raise FileNotFoundError(f"보정 파일을 찾을 수 없습니다: {path}. 먼저 calibrate_g2pp를 실행하세요.")