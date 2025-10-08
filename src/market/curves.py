# src/market/curves.py

import numpy as np
from math import log, exp
from scipy.interpolate import CubicSpline

def create_p_market_function(par_rates_pct: dict, freq: int = 1) -> callable:
    """
    파 스왑 금리로부터 부트스트래핑을 수행하여 할인 곡선 함수 P(t)를 생성합니다.
    """
    par_full = _fill_annual_par_curve_cubic(par_rates_pct)
    p_year = _bootstrap_discounts_from_par(par_full, freq)
    zero_curve = _create_continuous_zero_curve(p_year)
    
    p_market_func = lambda t: exp(-zero_curve(t) * t)
    return p_market_func

def _fill_annual_par_curve_cubic(par_rates_pct: dict) -> dict:
    """Cubic Spline을 이용해 비어있는 만기의 금리를 보간합니다."""
    known = sorted((int(T), v / 100.0) for T, v in par_rates_pct.items())
    known_T, known_S = [kv[0] for kv in known], [kv[1] for kv in known]
    maxT = known_T[-1]
    spline = CubicSpline(known_T, known_S, bc_type='natural')
    return dict(zip(range(1, maxT + 1), spline(range(1, maxT + 1))))

def _bootstrap_discounts_from_par(par_rates_dec: dict, freq: int) -> dict:
    """파 스왑 금리로부터 할인율(P)을 순차적으로 계산합니다."""
    p_year = {}
    delta = 1.0 / freq
    
    for T in sorted(par_rates_dec.keys()):
        S = par_rates_dec[T]
        coupon_times = [i * delta for i in range(1, int(round(T * freq)) + 1)]
        p_interpolator = _get_p_interpolator(p_year)
        
        known_sum = sum(p_interpolator(ti) for ti in coupon_times[:-1])
        p_T = (1.0 - S * delta * known_sum) / (1.0 + S * delta)
        p_year[T] = p_T
        
    return p_year

def _get_p_interpolator(p_year: dict) -> callable:
    """계산된 할인율 포인트를 보간하여 연속적인 함수를 만듭니다."""
    known_points = sorted([(0, 1.0)] + list(p_year.items()))
    if len(known_points) < 2: return lambda t: p_year.get(t, 0.0)
    
    known_T = [p[0] for p in known_points]
    known_lnP = [log(p[1]) for p in known_points]
    spline = CubicSpline(known_T, known_lnP, bc_type='natural')
    return lambda t: exp(spline(t))

def _create_continuous_zero_curve(p_year: dict) -> callable:
    """할인율로부터 연속적인 제로 금리 커브를 생성합니다."""
    maturities = np.array([0] + sorted(p_year.keys()))
    dfs = np.array([1.0] + [p_year[t] for t in sorted(p_year.keys())])
    
    # 0으로 나누기 방지
    safe_maturities = maturities + 1e-9
    
    zero_rates = -np.log(dfs) / safe_maturities
    zero_rates[0] = zero_rates[1] # t=0 시점 처리
    return CubicSpline(maturities, zero_rates)