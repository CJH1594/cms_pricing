# -*- coding: utf-8 -*-
"""
개선된 ATM 스왑션 변동성 표면 리스케일링 기능

주요 기능:
- Term structure를 고려한 만기별 스케일링
- Tenor별 차등 조정
- 다중 앵커 포인트(시장 데이터)를 사용한 표면 보정
- yfinance를 통한 MOVE 지수 다운로드 및 강화된 오류 처리
"""

import os
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from scipy.interpolate import Rbf

# cms_pricing 프로젝트의 설정 파일 임포트
# (프로젝트 루트가 Python 경로에 포함되어 있어야 합니다)
from config import settings

warnings.filterwarnings('ignore')


def to_years(label: str | int) -> float:
    """
    만기/테너 라벨(예: "1M", "2Y", 10)을 연 단위 숫자로 변환합니다.

    Args:
        label (str | int): 변환할 라벨.

    Returns:
        float: 연 단위 수치.

    Raises:
        ValueError: 인식할 수 없는 라벨 형식일 경우.
    """
    if isinstance(label, (int, float)):
        return float(label)
    
    label_str = str(label).strip().upper()
    if label_str.endswith("M"):
        return int(label_str[:-1]) / 12.0
    if label_str.endswith("Y"):
        return float(label_str[:-1])
    
    raise ValueError(f"알 수 없는 형식의 라벨입니다: {label}")


def _fetch_move_index(start_date: pd.Timestamp, end_date: pd.Timestamp, max_retries: int = 3) -> pd.Series | None:
    """
    Yahoo Finance에서 MOVE 지수를 다운로드합니다. (재시도 및 오류 처리 기능 포함)
    
    Args:
        start_date (pd.Timestamp): 조회 시작일.
        end_date (pd.Timestamp): 조회 종료일.
        max_retries (int): 실패 시 재시도 횟수.

    Returns:
        pd.Series | None: 성공 시 MOVE 지수 시계열, 실패 시 None.
    """
    for attempt in range(max_retries):
        try:
            # 영업일 기준 20일 전부터 5일 후까지 여유롭게 다운로드
            buffer_start = start_date - BDay(20)
            buffer_end = end_date + BDay(5)

            df = yf.download("^MOVE", start=buffer_start, end=buffer_end, auto_adjust=False, progress=False)

            if df is None or df.empty:
                raise IOError("MOVE 지수 다운로드 결과가 비어있습니다.")

            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[col].dropna()
            
            # 일별 마지막 값만 사용하도록 그룹화
            series = series.groupby(series.index.date).last()
            series.index = pd.to_datetime(series.index)
            
            if len(series) < 2:
                raise IOError(f"MOVE 데이터가 너무 적습니다 (포인트 수: {len(series)}개).")
            
            return series

        except Exception as e:
            print(f"  [경고] MOVE 지수 다운로드 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt >= max_retries - 1:
                return None
    return None

def _get_move_for_date(series: pd.Series, target_date: pd.Timestamp, max_lookback_days: int = 10) -> float:
    """
    다운로드된 시계열 데이터에서 특정 날짜의 MOVE 지수 값을 찾습니다.
    해당 날짜에 값이 없으면 가장 가까운 과거 영업일의 값을 사용합니다.
    
    Args:
        series (pd.Series): MOVE 지수 시계열.
        target_date (pd.Timestamp): 값을 찾을 목표 날짜.
        max_lookback_days (int): 최대 며칠 전 데이터까지 유효하게 볼 것인지 설정.

    Returns:
        float: 해당 날짜의 MOVE 지수 값.

    Raises:
        ValueError: 유효한 기간 내에 데이터를 찾지 못했을 경우.
    """
    s_clean = series.dropna()
    if target_date in s_clean.index:
        return float(s_clean.loc[target_date])

    # target_date보다 작거나 같은 인덱스 중 가장 최근 날짜를 찾음
    prev_indices = s_clean.index[s_clean.index <= target_date]
    if len(prev_indices) == 0:
        raise ValueError(f"{target_date.date()} 이전의 MOVE 데이터를 찾을 수 없습니다.")

    nearest_date = prev_indices.max()
    days_diff = (target_date - nearest_date).days

    if days_diff > max_lookback_days:
        print(f"  [경고] 목표일로부터 {days_diff}일 전의 MOVE 데이터를 사용합니다 ({nearest_date.date()}).")

    return float(s_clean.loc[nearest_date])


# volatility/rescaling.py 파일의 rescale_vol_surface 함수를 아래 코드로 교체하세요.

def rescale_vol_surface(
    base_surface: pd.DataFrame,
    old_date_str: str,                 # <<< 1. 인자 이름 변경 (base_date_str -> old_date_str)
    target_date_str: str,
    use_term_structure: bool = True,
    use_tenor_adjustment: bool = True,
    use_anchors: bool = True,
    verbose: bool = False              # <<< 2. verbose 인자 추가
) -> pd.DataFrame:
    """
    과거 스왑션 변동성 표면을 목표 날짜 기준으로 리스케일링합니다.

    Args:
        base_surface (pd.DataFrame): 리스케일링할 기준 변동성 표면.
        old_date_str (str): 기준 표면의 날짜 (YYYY-MM-DD).
        target_date_str (str): 리스케일링할 목표 날짜 (YYYY-MM-DD).
        use_term_structure (bool): 만기별 차등 스케일링 적용 여부.
        use_tenor_adjustment (bool): 테너별 차등 조정 적용 여부.
        use_anchors (bool): 앵커 포인트를 사용한 표면 보정 적용 여부.
        verbose (bool): True일 경우 상세 진행 과정을 출력합니다.

    Returns:
        pd.DataFrame: 리스케일링이 완료된 새로운 변동성 표면.
    """
    if verbose:
        print(f"'{old_date_str}'의 변동성 표면을 '{target_date_str}' 기준으로 리스케일링합니다.")

    # 1. 날짜 및 MOVE 지수 준비
    base_date = pd.Timestamp(old_date_str)
    target_date = pd.Timestamp(target_date_str)

    move_series = _fetch_move_index(base_date, target_date)
    move_base, move_target = None, None

    if move_series is not None:
        try:
            move_base = _get_move_for_date(move_series, base_date)
            move_target = _get_move_for_date(move_series, target_date)
            if verbose:
                print(f"✓ MOVE 지수 확인: {base_date.date()} ({move_base:.2f}), {target_date.date()} ({move_target:.2f})")
        except ValueError as e:
            if verbose:
                print(f"  [경고] MOVE 지수 값 추출 실패: {e}. 폴백 값을 확인합니다.")

    if move_base is None:
        move_base = settings.MOVE_FALLBACK_VALUES.get(old_date_str)
        if move_base and verbose:
            print(f"  [정보] 폴백 MOVE 값 사용 ({old_date_str}): {move_base}")
    if move_target is None:
        move_target = settings.MOVE_FALLBACK_VALUES.get(target_date_str)
        if move_target and verbose:
            print(f"  [정보] 폴백 MOVE 값 사용 ({target_date_str}): {move_target}")
            
    if move_base is None or move_target is None:
        raise ValueError("MOVE 지수를 확보할 수 없어 리스케일링을 진행할 수 없습니다.")
    
    k_base = move_target / move_base
    if verbose:
        print(f"→ 기본 스케일링 비율(k): {k_base:.4f}")

    # 2. 스케일링 행렬 생성
    scaled_surface = base_surface.copy()
    
    for expiry in base_surface.index:
        for tenor in base_surface.columns:
            k = k_base
            if use_term_structure:
                T = to_years(expiry)
                decay = settings.TERM_DECAY
                k = k ** (1 / (1 + decay * T))
            if use_tenor_adjustment:
                tenor_years = to_years(tenor)
                boost = settings.TENOR_SHORT_BOOST
                discount = settings.TENOR_LONG_DISCOUNT
                factor = 1.0
                if tenor_years <= 5:
                    factor = 1.0 + boost
                elif tenor_years > 10:
                    factor = 1.0 + discount
                else:
                    weight = (tenor_years - 5) / 5
                    factor = (1 + boost) * (1 - weight) + (1 + discount) * weight
                k *= factor
            scaled_surface.loc[expiry, tenor] *= k

    # 3. 앵커 포인트 보정
    anchor_points = settings.ANCHOR_POINTS
    if not (use_anchors and anchor_points):
        return scaled_surface

    if verbose:
        print(f"\n{len(anchor_points)}개의 앵커 포인트를 사용하여 표면을 보정합니다.")
    
    anchor_coords_exp, anchor_coords_ten, anchor_ratios = [], [], []
    for (exp_label, ten_label), target_vol in anchor_points.items():
        if exp_label in scaled_surface.index and ten_label in scaled_surface.columns:
            model_vol = scaled_surface.loc[exp_label, ten_label]
            ratio = target_vol / model_vol
            anchor_coords_exp.append(to_years(exp_label))
            anchor_coords_ten.append(to_years(ten_label))
            anchor_ratios.append(ratio)
            if verbose:
                print(f"  - {exp_label}×{ten_label}: 모델({model_vol:.4f}) → 시장({target_vol:.4f}), 비율={ratio:.4f}")

    if len(anchor_ratios) < 2:
        if verbose:
            print("  [경고] 2D 보간을 위한 앵커 포인트가 부족하여 보정을 건너뜁니다.")
        return scaled_surface

    rbf_interp = Rbf(anchor_coords_exp, anchor_coords_ten, anchor_ratios, function='multiquadric', smooth=0.1)
    grid_exp_years = [to_years(e) for e in scaled_surface.index]
    grid_ten_years = [to_years(t) for t in scaled_surface.columns]
    tenor_grid, expiry_grid = np.meshgrid(grid_ten_years, grid_exp_years)
    adjustment_grid = rbf_interp(expiry_grid, tenor_grid)
    final_surface = scaled_surface * adjustment_grid
    
    return final_surface


def _get_surface_path(filename: str) -> str:
    """데이터 디렉터리와 파일명을 결합하여 전체 파일 경로를 반환합니다."""
    return os.path.join(settings.DATA_DIR, filename)

def save_vol_surface(
    vol_surface: pd.DataFrame, 
    filename: str = settings.VOLATILITY_SURFACE_FILE
) -> None:
    """
    변동성 표면 DataFrame을 지정된 구조의 JSON 파일로 저장합니다.

    Args:
        vol_surface (pd.DataFrame): 저장할 변동성 표면.
        filename (str): 저장할 파일 이름. settings.py의 기본값을 사용합니다.
    """
    # 데이터 저장 디렉터리가 없으면 생성
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    # 전체 저장 경로 생성
    path = _get_surface_path(filename)
    
    print(f"\n계산된 변동성 표면을 다음 경로에 저장합니다:\n→ {path}")
    
    # 제공된 형식에 맞춰 데이터 구조화
    data = {
        "index": list(map(str, vol_surface.index)),
        "columns": list(map(str, vol_surface.columns)),
        "values": vol_surface.values.tolist(),
    }
    
    # JSON 파일로 저장
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("✓ 저장이 완료되었습니다.")