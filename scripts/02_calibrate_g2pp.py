# cms_pricing/scripts/02_calibrate_g2pp.py

import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import json

# --- 1. PYTHONPATH 설정 ---
# 프로젝트 최상위 폴더(cms_pricing의 부모)를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# --- 2. 절대 경로로 모듈 임포트 ---
from cms_pricing.config import settings
from cms_pricing.src.volatility import rescale_vol_surface # save/load는 rescale 함수가 담당
from cms_pricing.src.models import calibrate_g2pp, load_calibrated_params


# --- 3. 나머지 스크립트 로직 ---
def load_base_vol_surface(date_str: str) -> pd.DataFrame:
    """과거 기준일의 변동성 데이터를 로드하는 임시 함수."""
    if date_str == "2018-04-16":
        data_pct = [
            [12.4559, 15.6906, 18.5884, 18.7795, 18.0630, 17.7101, 17.5578],
            [14.1064, 17.1848, 19.5244, 19.6338, 18.9183, 18.5667, 18.4021],
            [15.3294, 18.1170, 20.7427, 20.7399, 19.9287, 19.5322, 19.3833],
            [17.7833, 20.1802, 22.1637, 21.9932, 21.0658, 20.5908, 20.3785],
            [22.1572, 23.2539, 23.9005, 23.0334, 21.9114, 21.2860, 21.0047],
        ]
        tenors_str = [f"{t}Y" for t in settings.TENORS]
        return pd.DataFrame(
            data=np.array(data_pct) / 100.0,
            index=settings.EXPIRY_LABELS,
            columns=tenors_str
        )
    else:
        raise FileNotFoundError(f"{date_str}에 해당하는 데이터가 없습니다.")

def main():
    """G2++ 모델 보정 프로세스를 실행하는 메인 함수"""
    print("=" * 70)
    print("G2++ 모델 보정 스크립트 시작")
    print("=" * 70)

    # 1. 기준 데이터 로드
    base_date_str = "2018-04-16"
    base_vol_surface = load_base_vol_surface(date_str=base_date_str)
    print(f"✓ '{base_date_str}' 기준 변동성 표면 로드 완료.")

    # 2. 목표 날짜 설정
    target_date_str = settings.TARGET_DATE_STR or date.today().strftime('%Y-%m-%d')
    print(f"→ 목표 날짜: {target_date_str}")

    # 3. 변동성 표면 리스케일링 또는 로드
    print("\n[단계 1] 변동성 표면 리스케일링 또는 로드 중...")
    vol_rescaled = rescale_vol_surface(
        base_surface=base_vol_surface,
        old_date_str=base_date_str,
        target_date_str=target_date_str,
        verbose=True,
    )
    print("✓ 리스케일링/로드 완료.")

    # 4. G2++ 모델 보정
    print("\n[단계 2] G2++ 모델 보정 중...")
    try:
        params = load_calibrated_params()
        print("✓ 저장된 파라미터를 로드했습니다. 보정을 건너뜁니다.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("  - 저장된 파라미터 없음. 새로 보정을 시작합니다.")
        from scipy.interpolate import CubicSpline
        from math import exp
        
        par_rates_dec = {t: r/100.0 for t, r in settings.PAR_RATES_PCT.items()}
        maturities = np.array([0] + sorted(par_rates_dec.keys()))
        dfs = np.array([1.0] + [1/(1+par_rates_dec[t])**t for t in sorted(par_rates_dec.keys())])
        zero_rates = -np.log(dfs) / (maturities + 1e-9)
        zero_rates[0] = zero_rates[1]
        continuous_zero_curve = CubicSpline(maturities, zero_rates)
        p_market_func = lambda t: exp(-continuous_zero_curve(t) * t)

        surface_pct = (vol_rescaled * 100.0).values.tolist()
        params = calibrate_g2pp(
            p_market_func=p_market_func,
            surface_pct=surface_pct,
            expiry_labels=settings.EXPIRY_LABELS,
            tenors=settings.TENORS,
            initial_params=settings.INITIAL_G2_PARAMS
        )

    print("\n[결과] 최종 G2++ 파라미터:")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  - {key:<18}: {value:.6f}")
        else:
            print(f"  - {key:<18}: {value}")
            
    print("\n" + "=" * 70)
    print("스크립트 실행 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()