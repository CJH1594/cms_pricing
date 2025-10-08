# cms_pricing/scripts/02_calibrate_g2pp.py

import sys
import os
import pandas as pd
import numpy as np
from datetime import date
import json

# --- 1. PYTHONPATH 설정 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- 2. 절대 경로로 모듈 임포트 ---
from cms_pricing.config import settings
from cms_pricing.src.volatility import rescale_vol_surface
from cms_pricing.src.models import calibrate_g2pp, load_calibrated_params
from cms_pricing.src.market.curves import create_p_market_function

# --- 3. main

def main():
    """G2++ 모델 보정 프로세스를 실행하는 메인 함수"""
    print("=" * 70)
    print("G2++ 모델 보정 스크립트 시작")
    print("=" * 70)

    # 1. 목표 날짜 설정
    # 벤치마크 재현 시에는 리스케일링이 불필요하지만, 로직의 일관성을 위해 유지합니다.
    # rescale 함수가 파일 존재를 감지하고 계산 없이 바로 로드할 것입니다.
    base_date_str = "2018-04-16" # 임의의 과거 날짜
    target_date_str = settings.TARGET_DATE_STR or date.today().strftime('%Y-%m-%d')
    print(f"→ 목표 날짜: {target_date_str}")

    # 2. 변동성 표면 리스케일링 또는 로드
    # rescale_vol_surface 함수가 data/volatility_surface.json 존재를 확인하고 바로 로드합니다.
    print("\n[단계 1] 변동성 표면 로드 중...")
    vol_surface = rescale_vol_surface(
        old_date_str=base_date_str,
        target_date_str=target_date_str,
        verbose=True,
    )
    print("✓ 표면 로드 완료.")

    # 3. G2++ 모델 보정
    print("\n[단계 2] G2++ 모델 보정 중...")
    try:
        params = load_calibrated_params()
        print("✓ 저장된 파라미터를 로드했습니다. 보정을 건너뜁니다.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("  - 저장된 파라미터 없음. 새로 보정을 시작합니다.")
        p_market_func = create_p_market_function(
            par_rates_pct=settings.PAR_RATES_PCT,
            freq=settings.FREQ
        )

        surface_pct = (vol_surface * 100.0).values.tolist()
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