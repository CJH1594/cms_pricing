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

# 2018-04-16 기준 과거 ATM 스왑션 변동성 표면 (단위: %)
HISTORICAL_VOL_SURFACE_PCT = [
    [12.4559, 15.6906, 18.5884, 18.7795, 18.0630, 17.7101, 17.5578],  # 1M
    [14.1064, 17.1848, 19.5244, 19.6338, 18.9183, 18.5667, 18.4021],  # 3M
    [15.3294, 18.1170, 20.7427, 20.7399, 19.9287, 19.5322, 19.3833],  # 6M
    [17.7833, 20.1802, 22.1637, 21.9932, 21.0658, 20.5908, 20.3785],  # 1Y
    [22.1572, 23.2539, 23.9005, 23.0334, 21.9114, 21.2860, 21.0047],  # 2Y
]

## 보정 초기값
INITIAL_G2_PARAMS = {
    "a": 0.268, "b": 0.337, "sigma": 0.0179, "eta": 0.0144, "rho": 0.977,
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