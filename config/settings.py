# -*- coding: utf-8 -*-
"""
프로젝트 전반에서 사용하는 기본 설정을 정의합니다.

• FREQ: 이자 지급 빈도 (예: 연 1회=1, 연 2회=2 등)
• 데이터 파일 경로: 부트스트랩 결과, 보정된 파라미터,
  변동성 표면 등을 JSON 형식으로 저장하기 위한 경로를 정의합니다.
• 기본 Par 금리: 금리 부트스트래핑에 사용될 스왑 시장의 Par 금리를 지정합니다.
"""

import os

## 프로젝트 루트 디렉터리 계산
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

## 데이터가 저장될 디렉터리
DATA_DIR = os.path.join(BASE_DIR, "data")

## 스크립트 디렉터리
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

## 파일 이름 정의
MARKET_DATA_FILE = "market_data.json"  # 부트스트랩 결과 저장
CALIBRATED_PARAMS_FILE = "calibrated_params.json"  # G2++ 보정 결과 저장
VOLATILITY_SURFACE_FILE = "volatility_surface.json"  # 리스케일된 변동성 표면 저장

## 기본 설정
FREQ = 1  # 고정 다리 지급 빈도 (연 1회)

## Par 스왑 금리 (단위: %). 필요한 경우 수정하거나 스크립트에서 로드 가능
PAR_RATES_PCT = {
    1: 3.649,
    2: 3.395,
    3: 3.335,
    5: 3.358,
    7: 3.452,
    10: 3.607,
    15: 3.810,
    30: 3.863,
}

## 스왑션 변동성 표면 기본 라벨 (만기 × 텐서)
EXPIRY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
TENORS = [1, 2, 5, 10, 15, 20, 30]

## 보정 초기값 (옵션). 원하는 경우 이 값을 변경하여 탐색을 다양화할 수 있습니다.
INITIAL_G2_PARAMS = {
    "a": 0.268,
    "b": 0.337,
    "sigma": 0.0179,
    "eta": 0.0144,
    "rho": 0.977,
}

## MOVE 지수 스케일링 관련 기본 파라미터
TERM_DECAY = 0.3
TENOR_SHORT_BOOST = 0.05
TENOR_LONG_DISCOUNT = -0.03

## 앵커 포인트 예시 (필요 시 수정). 키는 (만기, 텐서), 값은 목표 변동성 (소수)
ANCHOR_POINTS = {
    # ("1Y", "10Y"): 0.215,
}