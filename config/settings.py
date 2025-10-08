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
# 고정 다리 지급 빈도 (연 2회 = 반기)
FREQ = 2

## Par 스왑 금리 (단위: %).
PAR_RATES_PCT = {
    1: 3.649, 2: 3.395, 3: 3.335, 5: 3.358, 7: 3.452,
    10: 3.607, 15: 3.810, 30: 3.863,
}

## 스왑션 변동성 표면 기본 라벨 (만기 × 테너)
EXPIRY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
TENORS = [1, 2, 5, 10, 15, 20, 30]

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

## 타겟 날짜 설정 (YYYY-MM-DD 형식)
TARGET_DATE_STR = "2025-10-08"

# 네트워크가 제한된 환경에서 MOVE 지수 다운로드 실패 시 사용할 폴백 값들
# 키는 날짜 문자열, 값은 MOVE 지수 수치
MOVE_FALLBACK_VALUES = {
    # Monolith 예시와 일치하도록 설정
    "2018-04-16": 49.74,
    "2025-10-08": 75.43,
}

## 상품 정의
PRODUCT_DEFINITION = {
    'expiry': 1.0,
    'tenor_long': 10.0,
    'tenor_short': 2.0,
    'strike': 0.0055,
    'coupon': 1.0,
    'notional': 1.0,
}

## 몬테카를로 시뮬레이션 경로 수
NUM_PATHS = 50000

## 몬테카를로 시뮬레이션 스텝 수 (연간)
SIMULATION_NUM_STEPS_PER_YEAR = 252

## 가격 결정 결과 파일 이름
PRICING_RESULTS_FILE = "pricing_results.json"

## 리스크 분석 쇼크 값
RISK_SHOCK_AMOUNT = 0.0001 # 1bp