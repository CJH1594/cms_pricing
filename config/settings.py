# cms_pricing/config/settings.py

import os

## 프로젝트 루트 디렉터리 계산
# 이 파일(settings.py)이 있는 config 폴더의 부모 폴더(cms_pricing)가 기준
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

## 데이터가 저장될 디렉터리
DATA_DIR = os.path.join(BASE_DIR, "data")

## 스크립트 디렉터리
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

## 파일 이름 정의
MARKET_DATA_FILE = "market_data.json"
CALIBRATED_PARAMS_FILE = "calibrated_params.json"
VOLATILITY_SURFACE_FILE = "volatility_surface.json"

## 기본 설정
FREQ = 1

## Par 스왑 금리 (단위: %).
PAR_RATES_PCT = {
    1: 3.649, 2: 3.395, 3: 3.335, 5: 3.358, 7: 3.452,
    10: 3.607, 15: 3.810, 30: 3.863,
}

## 스왑션 변동성 표면 기본 라벨 (만기 × 테너)
EXPIRY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
TENORS = [1, 2, 5, 10, 15, 20, 30]

# --- 이 부분이 추가되었습니다 ---
## 기본 ATM 스왑션 변동성 표면 (단위: %). EXPIRY_LABELS x TENORS 크기
## (실제 시장 데이터로 교체해야 하는 예시 값입니다)
SWAPTION_VOL_SURFACE = [
    [35.5, 34.0, 31.5, 28.0, 26.5, 25.0, 23.5],  # 1M
    [36.0, 34.5, 32.0, 28.5, 27.0, 25.5, 24.0],  # 3M
    [35.0, 33.5, 31.0, 27.5, 26.0, 24.5, 23.0],  # 6M
    [32.0, 30.5, 28.5, 25.5, 24.5, 23.5, 22.0],  # 1Y
    [28.0, 27.0, 25.5, 23.5, 22.5, 21.5, 20.5],  # 2Y
]
# ---------------------------------

## 보정 초기값
INITIAL_G2_PARAMS = {
    "a": 0.1, "b": 0.2, "sigma": 0.01, "eta": 0.015, "rho": -0.5,
}

## MOVE 지수 스케일링 관련 기본 파라미터
TERM_DECAY = 0.3
TENOR_SHORT_BOOST = 0.05
TENOR_LONG_DISCOUNT = -0.03

## 앵커 포인트 예시
ANCHOR_POINTS = {}