#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_calibrate_g2pp.py
====================

이 스크립트는 과거 ATM 스왑션 변동성 표면을 MOVE 지수로 리스케일링한 후,
G2++ 모델의 파라미터를 최적화합니다. 최적화된 파라미터는
`data/calibrated_params.json`에 저장됩니다.
"""

import os
import sys
import datetime as _dt
import numpy as np
import pandas as pd

# PYTHONPATH 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cms_pricing.src.market import bootstrap_if_needed, create_continuous_zero_curve
from cms_pricing.src.volatility import rescale_vol_surface, save_vol_surface
from cms_pricing.src.models import calibrate_g2pp
from cms_pricing.config.settings import (
    EXPIRY_LABELS,
    TENORS,
)


def main() -> None:
    """
    스크립트 메인 함수.

    시장 데이터 부트스트랩, 변동성 표면 리스케일링, G2++ 보정을 순차적으로 수행합니다.
    이미 보정된 파라미터가 데이터 폴더에 존재하면 재사용하여 시간을 절약합니다.
    """
    # 1. 시장 데이터 부트스트랩 (필요 시 로드)
    par_full, P_year = bootstrap_if_needed()
    zero_curve = create_continuous_zero_curve(P_year)
    P_market = lambda t: np.exp(-zero_curve(t) * t)

    # 2. 과거 ATM 변동성 표면 정의 (2018-04-16, Black %)
    expiries = ["1M", "3M", "6M", "1Y", "2Y"]
    tenors = ["1Y", "2Y", "5Y", "10Y", "15Y", "20Y", "30Y"]
    data_2018_04_16_pct = [
        [12.4559, 15.6906, 18.5884, 18.7795, 18.0630, 17.7101, 17.5578],
        [14.1064, 17.1848, 19.5244, 19.6338, 18.9183, 18.5667, 18.4021],
        [15.3294, 18.1170, 20.7427, 20.7399, 19.9287, 19.5322, 19.3833],
        [17.7833, 20.1802, 22.1637, 21.9932, 21.0658, 20.5908, 20.3785],
        [22.1572, 23.2539, 23.9005, 23.0334, 21.9114, 21.2860, 21.0047],
    ]
    vol_old = pd.DataFrame(np.array(data_2018_04_16_pct) / 100.0, index=expiries, columns=tenors)

    # 3. 날짜 설정
    old_date_str = "2018-04-16"
    target_date_str = _dt.date.today().isoformat()

    # 4. MOVE 지수로 리스케일링
    vol_rescaled = rescale_vol_surface(vol_old, old_date_str=old_date_str, target_date_str=target_date_str)
    save_vol_surface(vol_rescaled)
    print("✓ 변동성 표면 리스케일링 및 저장 완료")

    # 5. G2++ 파라미터 보정 또는 로드
    # JSON 파일이 존재하면 로드하고, 없으면 보정 수행
    try:
        from cms_pricing.src.models import load_calibrated_params
        params = load_calibrated_params()
        print("✓ 저장된 파라미터를 로드했습니다. 보정을 건너뜁니다.")
    except Exception:
        # 보정 수행: surface_pct는 % 단위 변동성
        surface_pct = (vol_rescaled * 100.0).values.tolist()
        params = calibrate_g2pp(surface_pct, expiry_labels=EXPIRY_LABELS, tenors=TENORS, P_market=P_market)
        print("\n보정 결과:")
        for key in ['a', 'b', 'sigma', 'eta', 'rho']:
            print(f"  {key}: {params[key]:.6f}")
        print(f"  최종 오차: {params['final_error']:.8f}")
        print(f"  반복 횟수: {params['nit']}")
        print("\n✓ 보정된 파라미터가 data/에 저장되었습니다.")


if __name__ == "__main__":
    main()