#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_bootstrap_curve.py
=====================

이 스크립트는 시장 Par 금리를 이용하여 할인계수 P(0,T)을 부트스트래핑합니다.
결과는 `data/market_data.json`에 저장됩니다. 이미 계산된 값이 존재하면
파일을 로드합니다.
"""

import os
import sys
import numpy as np # Import numpy at the top
from pprint import pprint

# 프로젝트 루트 경로를 PYTHONPATH에 추가하여 cms_pricing 패키지를 찾을 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cms_pricing.src.market import bootstrap_if_needed, create_continuous_zero_curve
from cms_pricing.config.settings import FREQ


def main() -> None:
    # 부트스트랩 수행 또는 기존 파일 로드
    par_full, P_year = bootstrap_if_needed()
    # 제로 레이트 곡선 생성
    zero_curve = create_continuous_zero_curve(P_year)
    maxT = max(P_year.keys())
    print("부트스트랩 결과:")
    for T in sorted(P_year.keys()):
        PT = P_year[T]
        y_simple = PT ** (-1.0 / T) - 1.0
        z_cont = -np.log(PT) / T
        print(f"  T={T:2d}년 | P(0,T)={PT:.6f} | y_simple={y_simple*100:.3f}% | z_cont={z_cont*100:.3f}%")
    # 샘플 제로 레이트 출력
    print("\n샘플 제로 레이트 (연속복리):")
    for T in [1, 5, 10, maxT]:
        print(f"  T={T:2d}년 | z(T)={zero_curve(T)*100:.3f}%")
    print("\n✓ 할인계수 및 제로 레이트 곡선이 data/에 저장되었습니다.")


if __name__ == "__main__":
    main()