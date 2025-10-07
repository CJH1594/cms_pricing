# -*- coding: utf-8 -*-
"""
시장 금리 부트스트래핑 모듈
============================

이 모듈은 스왑 Par 금리를 사용하여 할인계수 P(0,T)을 부트스트래핑하는
함수들을 제공합니다. 결과는 JSON 파일로 저장하여 반복 실행 시
시간을 절약할 수 있습니다.

주요 함수
---------

* ``fill_annual_par_curve_cubic``: 누락된 연도 금리를 Cubic Spline으로 보간합니다.
* ``make_coupon_times``: 스왑 쿠폰 지급시점을 생성합니다.
* ``get_P_interpolator``: 이미 계산된 할인계수를 이용해 보간함수를 생성합니다.
* ``bootstrap_discounts_from_par``: Par 금리에서 할인계수를 계산합니다.
* ``create_continuous_zero_curve``: 연속복리 제로 레이트를 계산하는 스플라인을 생성합니다.
* ``save_market_data``/``load_market_data``: 계산 결과를 JSON 파일로 저장/로드합니다.
* ``bootstrap_if_needed``: 데이터가 존재하면 로드하고, 없으면 부트스트랩을 수행 후 저장합니다.
"""

import json
import os
from math import log, exp
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from ...config.settings import (
    DATA_DIR,
    MARKET_DATA_FILE,
    FREQ,
    PAR_RATES_PCT,
)


def fill_annual_par_curve_cubic(par_rates_pct: Dict[int, float]) -> Dict[int, float]:
    """주어진 Par 금리로부터 연간 금리를 Cubic Spline으로 보간합니다.

    인수
    ----
    par_rates_pct: dict
        키는 만기(년), 값은 퍼센트(%). 예: {1: 3.649, 2: 3.395, ...}

    반환값
    ------
    dict
        1년부터 최대 만기까지의 모든 정수 연도에 대한 금리(소수). 예: {1: 0.03649, 2: ...}
    """
    # 입력을 (만기, 금리) 튜플 리스트로 정렬
    known = sorted((int(T), v / 100.0) for T, v in par_rates_pct.items())
    known_T = [t for t, _ in known]
    known_S = [s for _, s in known]
    maxT = known_T[-1]

    # Cubic Spline으로 보간 (natural boundary 사용)
    spline = CubicSpline(known_T, known_S, bc_type="natural")
    all_T = range(1, maxT + 1)
    all_S = spline(all_T)
    return dict(zip(all_T, all_S))


def make_coupon_times(T: int, freq: int) -> Tuple[list, float]:
    """스왑의 쿠폰 지급 시점을 생성합니다.

    예를 들어 T=3, freq=2이면 [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]을 반환합니다.

    반환값은 (지급시점 리스트, 기간간격)입니다.
    """
    delta = 1.0 / freq
    n = int(round(T * freq))
    return [i * delta for i in range(1, n + 1)], delta


def get_P_interpolator(P_year: Dict[int, float]):
    """이미 계산된 할인계수(P_year)를 사용하여 보간함수를 생성합니다.

    P_year가 비어 있으면 1을 반환하는 간단한 딕셔너리 getter를 돌려줍니다.
    보간은 ln(P)에 대해 Cubic Spline을 적용하여 음이 아닌 할인계수를 보장합니다.
    """
    # (0,1) 포인트 포함
    known_points = sorted([(0, 1.0)] + list(P_year.items()))
    if len(known_points) < 2:
        # 아직 계산된 포인트가 거의 없는 경우, 간단한 getter 사용
        return lambda t: P_year.get(t, 1.0)

    known_T = [p[0] for p in known_points]
    known_lnP = [log(p[1]) for p in known_points]
    spline = CubicSpline(known_T, known_lnP, bc_type="natural")
    return lambda t: exp(float(spline(t)))


def bootstrap_discounts_from_par(par_rates_dec: Dict[int, float], freq: int = FREQ) -> Dict[int, float]:
    """Par 금리로부터 할인계수 P(0,T)을 부트스트랩합니다.

    구현은 고정 다리 지급을 고려한 스왑 밸류에이션 방식을 사용합니다.

    인수
    ----
    par_rates_dec: dict
        키는 만기(년), 값은 스왑 금리(소수)
    freq: int
        고정 다리 지급 빈도

    반환값
    ------
    dict
        만기별 할인계수 {T: P(0,T)}
    """
    P_year: Dict[int, float] = {}
    for T in sorted(par_rates_dec.keys()):
        S = par_rates_dec[T]
        coupon_times, delta = make_coupon_times(T, freq)
        P_interp = get_P_interpolator(P_year)

        # 마지막 지급시점 전까지의 할인계수 합
        known_sum = sum(P_interp(ti) for ti in coupon_times[:-1])
        numerator = 1.0 - S * delta * known_sum
        denom = 1.0 + S * delta
        P_T = numerator / denom
        P_year[T] = P_T
    return P_year


def create_continuous_zero_curve(P_year: Dict[int, float]):
    """할인계수로부터 연속복리 제로 레이트 곡선을 생성합니다.

    제로 레이트는 -ln(P)/T 로 정의합니다. T=0에서는 첫 번째 값을 복사합니다.
    CubicSpline을 사용하여 t ∈ [0, maxT] 구간의 매끄러운 스플라인을 제공합니다.
    """
    maturities = np.array([0] + sorted(P_year.keys()), dtype=float)
    dfs = np.array([1.0] + [P_year[t] for t in sorted(P_year.keys())], dtype=float)
    zero_rates = -np.log(dfs) / (maturities + 1e-9)
    # t=0에서는 첫 번째 양수 값으로 채워넣음
    zero_rates[0] = zero_rates[1]
    spline = CubicSpline(maturities, zero_rates, bc_type="natural")
    return spline


def _get_market_data_path(filename: str = MARKET_DATA_FILE) -> str:
    """내부적으로 데이터 파일의 절대 경로를 계산합니다."""
    return os.path.join(DATA_DIR, filename)


def save_market_data(par_full: Dict[int, float], P_year: Dict[int, float], filename: str = MARKET_DATA_FILE) -> None:
    """부트스트랩 결과를 JSON 파일로 저장합니다.

    JSON은 키를 문자열로 저장하므로, 추후 로드시 int로 변환합니다.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _get_market_data_path(filename)
    data = {
        "par_full": {str(k): v for k, v in par_full.items()},
        "P_year": {str(k): v for k, v in P_year.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_market_data(filename: str = MARKET_DATA_FILE) -> Tuple[Dict[int, float], Dict[int, float]]:
    """저장된 부트스트랩 결과를 로드합니다.

    파일이 없으면 FileNotFoundError를 발생시킵니다.
    """
    path = _get_market_data_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"시장 데이터 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    par_full = {int(k): float(v) for k, v in data["par_full"].items()}
    P_year = {int(k): float(v) for k, v in data["P_year"].items()}
    return par_full, P_year


def bootstrap_if_needed(par_rates_pct: Dict[int, float] = None,
                        freq: int = FREQ,
                        force: bool = False,
                        filename: str = MARKET_DATA_FILE) -> Tuple[Dict[int, float], Dict[int, float]]:
    """할인계수 부트스트래핑을 수행하거나, 기존 결과를 로드합니다.

    - 이미 저장된 결과가 있고 force=False이면 파일을 로드합니다.
    - force=True이면 새로 계산하고 파일을 덮어씁니다.

    인수
    ----
    par_rates_pct: dict, optional
        사용자 정의 Par 금리. 기본값은 settings.PAR_RATES_PCT를 사용합니다.
    freq: int
        고정 다리 지급 빈도
    force: bool
        True이면 항상 계산 후 저장, False이면 존재할 경우 로드
    filename: str
        저장/로드할 파일 이름

    반환값
    ------
    (par_full, P_year)
    """
    if par_rates_pct is None:
        par_rates_pct = PAR_RATES_PCT.copy()
    # 저장된 파일이 존재하고 재계산을 원하지 않으면 로드
    try:
        if not force:
            par_full, P_year = load_market_data(filename)
            return par_full, P_year
    except FileNotFoundError:
        # 파일이 없으면 새로 계산
        pass
    # Cubic Spline을 이용해 연간 Par 금리 보간
    par_full = fill_annual_par_curve_cubic(par_rates_pct)
    # 소수로 변환된 Par 금리를 사용하여 할인계수 부트스트랩
    P_year = bootstrap_discounts_from_par(par_full, freq=freq)
    # 결과 저장
    save_market_data(par_full, P_year, filename)
    return par_full, P_year