# -*- coding: utf-8 -*-
"""
G2++ 금리모형 및 시뮬레이션
============================

이 모듈은 G2++(G-Squared Plus Plus) 금리모형의 핵심 함수들을 제공합니다.

G2++ 모형은 두 개의 Ornstein-Uhlenbeck 프로세스 x(t), y(t)와 드리프트 함수
φ(t)로 단기 금리를 나타냅니다::

    r(t) = x(t) + y(t) + φ(t)
    dx = -a x dt + σ dW₁
    dy = -b y dt + η dW₂
    dW₁·dW₂ = ρ dt

여기에서 a, b는 평균회귀 속도, σ, η는 변동성, ρ는 상관계수입니다.

함수 설명
---------

* ``make_lnP_spline``: 할인계수 P(0,T)로부터 ln P 스플라인과 선도금리 f(0,t) 함수를 생성합니다.
* ``phi_g2pp_factory``: 주어진 파라미터에 대해 φ(t)와 f(0,t) 함수를 생성합니다.
* ``calculate_V``: V(t,T) 보조 함수. 이후의 할인계수 계산에 사용됩니다.
* ``B``: Brigo & Mercurio 공식의 B 함수.
* ``simulate_g2_paths``: 지정된 만기까지 G2++ 경로를 시뮬레이션하고 할인계수 ∫ r(s) ds를 계산합니다.
* ``calculate_forward_swap_rate``: 미래 시점 t에서 tenor년 만기 스왑의 선도 금리를 계산합니다.
"""

from functools import lru_cache
from math import exp, sqrt
from typing import Callable, Dict, Tuple

import numpy as np


def make_lnP_spline(P_year: Dict[int, float]) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """할인계수로부터 ln P 스플라인과 선도금리 f(0,t) 함수를 생성합니다.

    P_year: 만기(정수) → 할인계수 값
    반환: (ln P 스플라인 함수, f0 함수)
    f0(t) = -d/dt ln P(0,t)
    """
    from scipy.interpolate import CubicSpline
    # 0 시점 포함
    Ts = [0.0] + sorted(float(T) for T in P_year.keys())
    lnPs = [0.0] + [np.log(P_year[int(T)]) for T in sorted(P_year.keys())]
    spline_lnP = CubicSpline(Ts, lnPs, bc_type="natural")
    spline_dlnP = spline_lnP.derivative()

    def f0(t: float) -> float:
        # t=0에서는 우측 도함수를 근사
        if t < 1e-9:
            return -float(spline_dlnP(1e-6))
        return -float(spline_dlnP(t))

    return spline_lnP, f0


def phi_g2pp_factory(P_year: Dict[int, float],
                     a: float,
                     b: float,
                     sigma: float,
                     eta: float,
                     rho: float) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """G2++ 모델의 φ(t)와 선도금리 f(0,t) 함수를 생성합니다.

    φ(t) = f(0,t) + (σ²/2a²)(1-e^{-at})² + (η²/2b²)(1-e^{-bt})² + (ρση/ab)(1-e^{-at})(1-e^{-bt})
    """
    _, f0 = make_lnP_spline(P_year)

    def phi(t: float) -> float:
        # 0 근처에서 분모가 0이 되는 것을 방지
        if t < 1e-9:
            t = 1e-9
        ea = exp(-a * t)
        eb = exp(-b * t)
        
        # term_x calculation
        if abs(a) < 1e-9:
            term_x = 0.5 * (sigma * t)**2
        else:
            term_x = (sigma ** 2) / (2.0 * a ** 2) * (1.0 - ea) ** 2

        # term_y calculation
        if abs(b) < 1e-9:
            term_y = 0.5 * (eta * t)**2
        else:
            term_y = (eta ** 2) / (2.0 * b ** 2) * (1.0 - eb) ** 2

        # cross term calculation
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            cross = rho * sigma * eta * t**2
        elif abs(a) < 1e-9:
            cross = (rho * sigma * eta / b) * t * (1.0 - eb) # Corrected to use 't'
        elif abs(b) < 1e-9:
            cross = (rho * sigma * eta / a) * (1.0 - ea) * t # Corrected to use 't'
        else:
            cross = (rho * sigma * eta) / (a * b) * (1.0 - ea) * (1.0 - eb)

        return f0(t) + term_x + term_y + cross

    return phi, f0


# -----------------------------
# V(t,T) 캐시: dict → 튜플 키화
# -----------------------------

def _params_key(params: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    """캐시 키로 사용할 (a, b, sigma, eta, rho) 튜플 생성."""
    return (params["a"], params["b"], params["sigma"], params["eta"], params["rho"])


@lru_cache(maxsize=4096)
def _calculate_V_cached(t: float, T: float,
                        a: float, b: float, sigma: float, eta: float, rho: float) -> float:
    """Brigo & Mercurio의 V(t,T) 계산 (캐시용, 해시가능 인자만 받음)."""
    T_m_t = T - t
    if abs(a) < 1e-12 or abs(b) < 1e-12:
        return 0.0

    # term1 calculation
    if abs(a) < 1e-9:
        term1 = (sigma**2 / 2) * T_m_t**2
    else:
        term1 = (sigma**2 / a**2) * (T_m_t + (2 / a) * np.exp(-a * T_m_t) - (1 / (2 * a)) * np.exp(-2 * a * T_m_t) - 3 / (2 * a))

    # term2 calculation
    if abs(b) < 1e-9:
        term2 = (eta**2 / 2) * T_m_t**2
    else:
        term2 = (eta**2 / b**2) * (T_m_t + (2 / b) * np.exp(-b * T_m_t) - (1 / (2 * b)) * np.exp(-2 * b * T_m_t) - 3 / (2 * b))
    
    # term3 calculation
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        term3 = rho * sigma * eta * T_m_t**2
    elif abs(a) < 1e-9:
        term3 = (2 * rho * sigma * eta / b) * (T_m_t + (np.exp(-b * T_m_t) - 1) / b - (np.exp(-b * T_m_t) - 1) / b)
    elif abs(b) < 1e-9:
        term3 = (2 * rho * sigma * eta / a) * (T_m_t + (np.exp(-a * T_m_t) - 1) / a - (np.exp(-a * T_m_t) - 1) / a)
    else:
        term3 = (2 * rho * sigma * eta / (a * b)) * (T_m_t + (np.exp(-a * T_m_t) - 1) / a + (np.exp(-b * T_m_t) - 1) / b - (np.exp(-(a + b) * T_m_t) - 1) / (a + b))

    return float(term1 + term2 + term3)


def calculate_V(t: float, T: float, params: Dict[str, float]) -> float:
    """원래 시그니처 유지: dict를 받아 캐시 가능한 튜플로 변환해 호출."""
    a, b, sigma, eta, rho = _params_key(params)
    return _calculate_V_cached(t, T, a, b, sigma, eta, rho)


@lru_cache(maxsize=2048)
def B(z: float, t: float, T: float) -> float:
    """Brigo & Mercurio 공식의 B 함수: B_z(t,T) = (1 - e^{-z(T-t)})/z"""
    if abs(z) < 1e-12:
        return T - t
    return (1 - np.exp(-z * (T - t))) / z


def simulate_g2_paths(params: Dict[str, float],
                      T_expiry: float,
                      num_paths: int,
                      num_steps: int,
                      phi_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G2++ 경로를 시뮬레이션하여 (x_T, y_T, 할인계수) 배열을 반환합니다.

    - ``params``: a, b, sigma, eta, rho가 포함된 딕셔너리
    - ``T_expiry``: 시뮬레이션 만기 (년)
    - ``num_paths``: 시뮬레이션 경로 수
    - ``num_steps``: 시간 분할 개수
    - ``phi_func``: φ(t) 함수

    반환값은 (x_T, y_T, D_paths)이며, D_paths는 exp(-∫ r(s) ds)입니다.
    """
    a = params['a']; b = params['b']; sigma = params['sigma']; eta = params['eta']; rho = params['rho']
    dt = T_expiry / num_steps

    # 초기 상태 설정
    x = np.zeros(num_paths)
    y = np.zeros(num_paths)
    integ_r = np.zeros(num_paths)
    current_time = 0.0

    # 상관된 브라운 운동 샘플링
    for _ in range(num_steps):
        dw1 = np.random.normal(0.0, sqrt(dt), num_paths)
        dw2_indep = np.random.normal(0.0, sqrt(dt), num_paths)
        dw2 = rho * dw1 + sqrt(1.0 - rho ** 2) * dw2_indep

        # OU 프로세스 업데이트 (Euler–Maruyama)
        x += -a * x * dt + sigma * dw1
        y += -b * y * dt + eta * dw2

        # 시간 진행 및 금리 적분
        current_time += dt
        phi_t = phi_func(current_time)
        r_t = x + y + phi_t
        integ_r += r_t * dt

    D_paths = np.exp(-integ_r)
    return x, y, D_paths


def calculate_forward_swap_rate(t_future: float,
                                tenor: float,
                                x_t: float,
                                y_t: float,
                                params: Dict[str, float],
                                P_market: Callable[[float], float],
                                V_0_func: Callable[[float], float]) -> float:
    """t_future 시점에서 tenor년 만기 스왑의 선도 금리를 계산합니다.

    Brigo & Mercurio의 공식에 따라, P(t,T)를 A(t,T) * exp(-B_a(t,T) x - B_b(t,T) y)로
    계산하여 스왑 가격을 구한 뒤 선도 금리를 산출합니다. annuity가 0이면 0을 반환합니다.
    """
    a = params['a']; b = params['b']

    # 결제 시점 생성 (0.5년 간격)
    swap_start = t_future
    swap_end = t_future + tenor
    payment_times = np.arange(swap_start + 0.5, swap_end + 0.01, 0.5)

    # 내부 함수: t_future 시점에서 채권 가격 계산
    def get_future_P(T_maturity: float) -> float:
        # 시장 할인계수
        P_0_T = P_market(T_maturity)
        P_0_t = P_market(t_future)

        # 분산 항
        V_t_T = calculate_V(t_future, T_maturity, params)
        V_0_T = V_0_func(T_maturity)
        V_0_t = V_0_func(t_future)

        # A(t,T) = [P(0,T)/P(0,t)] * exp(0.5*(V(t,T) - V(0,T) + V(0,t)))
        A_t_T = (P_0_T / P_0_t) * exp(0.5 * (V_t_T - V_0_T + V_0_t))

        # B_a, B_b
        B_a = B(a, t_future, T_maturity)
        B_b = B(b, t_future, T_maturity)

        return A_t_T * exp(-B_a * x_t - B_b * y_t)

    # annuity 및 부유 leg PV
    annuity = sum(0.5 * get_future_P(ti) for ti in payment_times)
    floating_pv = get_future_P(swap_start) - get_future_P(swap_end)

    if annuity == 0:
        return 0.0
    return float(floating_pv / annuity)