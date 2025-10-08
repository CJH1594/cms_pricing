# cms_pricing/src/risk/pricing/utils.py

import json
import os

from ...config.settings import DATA_DIR, PRICING_RESULTS_FILE

# DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
# PRICING_RESULTS_FILE = os.path.join(DATA_DIR, 'pricing_results.json')

def save_pricing_results(results: dict) -> None:
    """
    가격 계산 결과를 JSON 파일로 저장합니다.
    """
    # 이미 data 폴더가 있다면 새로 만들 필요 없음 (exist_ok=True는 그대로 둬도 됨)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 반드시 data 디렉터리 밑에 저장되도록 경로를 합침
    pricing_results_file = os.path.join(DATA_DIR, "pricing_results.json")

    with open(pricing_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✓ 가격 계산 결과가 {pricing_results_file}에 저장되었습니다.")


def load_pricing_results() -> dict:
    """
    저장된 가격 계산 결과를 JSON 파일에서 불러옵니다.
    """
    if not os.path.exists(PRICING_RESULTS_FILE):
        raise FileNotFoundError(f"가격 계산 결과 파일이 없습니다: {PRICING_RESULTS_FILE}")
    with open(PRICING_RESULTS_FILE, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"가격 계산 결과 파일이 손상되었습니다: {PRICING_RESULTS_FILE}") from e
    print(f"✓ 가격 계산 결과가 {PRICING_RESULTS_FILE}에서 로드되었습니다.")
    return results
