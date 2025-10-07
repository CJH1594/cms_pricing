# -*- coding: utf-8 -*-
"""
스왑션 변동성 표면 리스케일링 모듈
==============================

이 모듈은 MOVE 지수 및 기타 조정 요인을 사용하여 과거 ATM 스왑션 변동성
표면을 현재 시점으로 리스케일링합니다. 또한 만기별/텐서별 차등 조정과
다중 앵커 포인트를 지원합니다. 계산된 표면은 JSON 파일로 저장하거나
로드할 수 있습니다.

사용 예시::

    from cms_pricing.src.volatility.rescaling import rescale_vol_surface, load_vol_surface
    vol_old = pd.DataFrame(...)
    vol_new = rescale_vol_surface(
        vol_old,
        old_date_str="2018-04-16",
        target_date_str="2025-10-07",
        use_term_structure=True,
        use_tenor_adjustment=True,
        anchor_points={}
    )
    # JSON으로 저장
    save_vol_surface(vol_new)

"""

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from ...config.settings import (
    DATA_DIR,
    VOLATILITY_SURFACE_FILE,
    TERM_DECAY,
    TENOR_SHORT_BOOST,
    TENOR_LONG_DISCOUNT,
    ANCHOR_POINTS,
)

try:
    import yfinance as yf  # 외부 호출이 불가한 환경에서는 실패할 수 있음
except ImportError:
    yf = None  # 테스트 환경에서 yfinance가 없을 경우 대비


def to_years(label: str) -> float:
    """만기나 텐서 라벨을 연 단위로 변환합니다.

    ``"6M" → 0.5``, ``"1Y" → 1.0``과 같이 숫자를 반환합니다.
    라벨이 잘못된 경우 ValueError를 발생시킵니다.
    """
    label = label.strip().upper()
    if label.endswith("M"):
        return int(label[:-1]) / 12.0
    if label.endswith("Y"):
        return int(label[:-1]) * 1.0
    raise ValueError(f"Unknown label format: {label}")


def fetch_move_series_robust(start_ts: pd.Timestamp,
                             end_ts: pd.Timestamp,
                             max_retries: int = 3) -> pd.Series:
    """MOVE 지수 시간 시리즈를 다운로드합니다.

    yfinance 라이브러리를 사용하여 MOVE(^MOVE) 지수를 다운로드합니다. 네트워크
    환경에 따라 일시적인 오류가 있을 수 있으므로 재시도 로직을 포함합니다.
    ``yfinance``가 설치되지 않은 경우 RuntimeError를 발생시킵니다.
    """
    if yf is None:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다. MOVE 데이터를 다운로드할 수 없습니다.")

    for attempt in range(max_retries):
        try:
            # 일정 버퍼를 두어 공휴일 및 데이터 누락에 대비
            buffer_start = start_ts - BDay(20)
            buffer_end = end_ts + BDay(5)
            df = yf.download("^MOVE", start=buffer_start, end=buffer_end,
                           auto_adjust=False, progress=False)
            if df is None or df.empty:
                raise RuntimeError("MOVE download returned empty data")
            # 주가 데이터에서 Adj Close 또는 Close 컬럼 추출
            if "Adj Close" in df.columns:
                col = "Adj Close"
            elif "Close" in df.columns:
                col = "Close"
            else:
                raise RuntimeError(f"예상되는 컬럼을 찾을 수 없습니다: {df.columns.tolist()}")
            s = df[col]
            s = s.groupby(s.index.date).last()
            s.index = pd.to_datetime(s.index)
            if len(s) < 5:
                raise RuntimeError(f"MOVE 데이터가 너무 적습니다: {len(s)} points")
            return s
        except Exception as e:
            if attempt < max_retries - 1:
                # 다음 시도에서 재시도
                continue
            raise RuntimeError(f"MOVE 데이터를 가져오는 데 실패했습니다: {e}")


def nearest_business_value(series: pd.Series,
                           ts: pd.Timestamp,
                           max_lookback_days: int = 10) -> float:
    """주어진 날짜 또는 그 이전 가장 가까운 영업일의 값을 반환합니다.

    시리즈의 인덱스는 날짜여야 하며, 데이터가 없는 경우 max_lookback_days
    이내의 날짜를 찾습니다. 데이터가 없으면 ValueError를 발생시킵니다.
    """
    s = series.dropna()
    if ts in s.index:
        # s.at[...]은 항상 스칼라를 반환하여 FutureWarning을 회피
        return float(s.at[ts])
    prev_indices = s.index[s.index <= ts]
    if len(prev_indices) == 0:
        raise ValueError(f"{ts.date()} 이전에 MOVE 데이터가 없습니다.")
    nearest_date = prev_indices.max()
    days_diff = (ts - nearest_date).days
    if days_diff > max_lookback_days:
        # 사용자가 경고를 보고 조정할 수 있도록 프린트
        print(f"  ⚠ Warning: Using MOVE data from {days_diff} days ago ({nearest_date.date()})")
    # s.at[...]은 스칼라 값을 반환하여 DeprecatedWarning을 회피
    return float(s.at[nearest_date])


def calculate_term_adjusted_k(expiry_label: str, k_base: float, decay: float = TERM_DECAY) -> float:
    """만기별 MOVE 비율 조정 계수를 계산합니다.

    만기가 길어질수록 MOVE 영향력을 감쇠시키기 위해 decay 파라미터를 사용합니다.
    """
    T = to_years(expiry_label)
    # 예: T=0일 때 k_base, T=2일 때 k_base^(1/(1+decay*T))
    adjusted_k = k_base ** (1.0 / (1.0 + decay * T))
    return adjusted_k


def calculate_tenor_adjustment_factor(tenor_label: str,
                                     short_boost: float = TENOR_SHORT_BOOST,
                                     long_discount: float = TENOR_LONG_DISCOUNT) -> float:
    """텐서별 추가 조정 계수를 계산합니다.

    단기 텐서(≤5Y)는 추가 승수를 적용하고, 장기 텐서(>10Y)는 감쇠시키며,
    그 사이 구간은 선형 보간합니다.
    """
    tenor_years = to_years(tenor_label)
    if tenor_years <= 5:
        return 1.0 + short_boost
    elif tenor_years > 10:
        return 1.0 + long_discount
    else:
        weight = (tenor_years - 5) / 5.0
        return (1.0 + short_boost) * (1.0 - weight) + (1.0 + long_discount) * weight


def create_scaling_matrix(expiries: list,
                         tenors: list,
                         k_base: float,
                         use_term: bool = True,
                         use_tenor: bool = True,
                         term_decay: float = TERM_DECAY,
                         tenor_short_boost: float = TENOR_SHORT_BOOST,
                         tenor_long_discount: float = TENOR_LONG_DISCOUNT) -> pd.DataFrame:
    """만기×텐서별 스케일링 행렬을 생성합니다.

    k_base는 기본 MOVE 비율이며, 만기별로 감쇠시키거나 텐서별로 조정합니다.
    """
    k_matrix = pd.DataFrame(index=expiries, columns=tenors, dtype=float)
    for exp in expiries:
        for ten in tenors:
            k = k_base
            if use_term:
                k = calculate_term_adjusted_k(exp, k, term_decay)
            if use_tenor:
                k *= calculate_tenor_adjustment_factor(ten, tenor_short_boost, tenor_long_discount)
            k_matrix.loc[exp, ten] = k
    return k_matrix


def apply_multiple_anchors(vol_scaled: pd.DataFrame,
                          anchor_points: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    """여러 앵커 포인트를 사용하여 스케일된 표면을 추가로 조정합니다.

    anchor_points는 {(만기, 텐서): 목표변동성} 형태로 지정합니다. 두 개 이상
    포인트가 있을 경우 2D RBF 보간을 사용하여 비율을 계산합니다. 앵커가
    없으면 입력 표면을 그대로 반환합니다.
    """
    if not anchor_points:
        return vol_scaled
    # 목표 대비 현재 스케일링 값의 비율 계산
    adjustment_ratios = {}
    for (exp, ten), target_vol in anchor_points.items():
        if exp in vol_scaled.index and ten in vol_scaled.columns:
            current_vol = vol_scaled.loc[exp, ten]
            ratio = target_vol / current_vol
            adjustment_ratios[(exp, ten)] = ratio
    if len(adjustment_ratios) == 0:
        return vol_scaled
    if len(adjustment_ratios) == 1:
        ratio = list(adjustment_ratios.values())[0]
        return vol_scaled * ratio
    # RBF 보간을 이용한 부드러운 비율 계산
    from scipy.interpolate import Rbf
    expiry_years = [to_years(e) for e in vol_scaled.index]
    tenor_years = [to_years(t) for t in vol_scaled.columns]
    anchor_exp_years = [to_years(e) for e, _ in adjustment_ratios.keys()]
    anchor_ten_years = [to_years(t) for _, t in adjustment_ratios.keys()]
    anchor_vals = list(adjustment_ratios.values())
    rbf = Rbf(anchor_exp_years, anchor_ten_years, anchor_vals,
              function='multiquadric', smooth=0.1)
    adjustment_grid = np.zeros_like(vol_scaled.values, dtype=float)
    for i, exp_y in enumerate(expiry_years):
        for j, ten_y in enumerate(tenor_years):
            adjustment_grid[i, j] = rbf(exp_y, ten_y)
    return vol_scaled * adjustment_grid


def rescale_vol_surface(vol_old: pd.DataFrame,
                        old_date_str: str,
                        target_date_str: str,
                        use_term_structure: bool = True,
                        use_tenor_adjustment: bool = True,
                        use_multiple_anchors: bool = False,
                        term_decay: float = TERM_DECAY,
                        tenor_short_boost: float = TENOR_SHORT_BOOST,
                        tenor_long_discount: float = TENOR_LONG_DISCOUNT,
                        anchor_points: Dict[Tuple[str, str], float] = None) -> pd.DataFrame:
    """MOVE 지수를 이용하여 과거 변동성 표면을 현재 시점으로 리스케일링합니다.

    vol_old는 DataFrame 형태로 입력되며, 인덱스는 만기 문자열,
    컬럼은 텐서(정수 또는 문자열)입니다. 리턴되는 DataFrame은 같은 형식으로
    현재 시점의 변동성을 제공합니다.
    """
    if anchor_points is None:
        anchor_points = ANCHOR_POINTS.copy()
    # 날짜 변환
    old_date = pd.Timestamp(old_date_str)
    target_date = pd.Timestamp(target_date_str)
    # MOVE 지수 다운로드 및 비율 계산
    try:
        move_series = fetch_move_series_robust(old_date, target_date)
        move_old = nearest_business_value(move_series, old_date)
        move_new = nearest_business_value(move_series, target_date)
        k_base = move_new / move_old
    except Exception as e:
        # yfinance가 없거나 네트워크 오류가 있는 경우 원본 표면 사용
        print(f"⚠ MOVE 지수 데이터를 가져오는 데 실패했습니다: {e}. 기본 스케일 1.0을 사용합니다.")
        k_base = 1.0
    # 스케일링 행렬 생성
    expiries = list(vol_old.index)
    tenors = list(vol_old.columns)
    k_matrix = create_scaling_matrix(
        expiries,
        tenors,
        k_base,
        use_term=use_term_structure,
        use_tenor=use_tenor_adjustment,
        term_decay=term_decay,
        tenor_short_boost=tenor_short_boost,
        tenor_long_discount=tenor_long_discount,
    )
    vol_scaled = vol_old * k_matrix
    if use_multiple_anchors and anchor_points:
        vol_scaled = apply_multiple_anchors(vol_scaled, anchor_points)
    return vol_scaled


def _get_surface_path(filename: str = VOLATILITY_SURFACE_FILE) -> str:
    """변동성 표면 파일의 절대경로를 반환합니다."""
    return os.path.join(DATA_DIR, filename)


def save_vol_surface(vol_surface: pd.DataFrame, filename: str = VOLATILITY_SURFACE_FILE) -> None:
    """변동성 표면을 JSON 파일로 저장합니다.

    DataFrame의 인덱스와 컬럼을 문자열로 변환하여 딕셔너리로 저장합니다.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _get_surface_path(filename)
    data = {
        "index": list(map(str, vol_surface.index)),
        "columns": list(map(str, vol_surface.columns)),
        "values": vol_surface.values.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_vol_surface(filename: str = VOLATILITY_SURFACE_FILE) -> pd.DataFrame:
    """저장된 변동성 표면을 로드하여 DataFrame으로 반환합니다.

    파일이 없으면 FileNotFoundError를 발생시킵니다.
    """
    path = _get_surface_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"변동성 표면 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = [str(x) for x in data["index"]]
    columns = [str(x) for x in data["columns"]]
    values = np.array(data["values"], dtype=float)
    return pd.DataFrame(values, index=index, columns=columns)