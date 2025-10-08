# scripts/04_risk_analysis.py

import os
import sys

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cms_pricing.src.models import get_analytic_model
from cms_pricing.src.risk import run_risk_analysis

def main():
    """
    선택된 가격결정 모델을 사용하여 리스크 분석을 실행한다.
    아래 MODELS_TO_RUN 리스트를 수정하여 여러 모델을 한번에 분석할 수 있다.
    """
    # --- 분석할 모델 선택 ---
    MODELS_TO_RUN = ["bachelier", "gbm"]  # "bachelier", "gbm" 중 원하는 모델을 리스트에 추가
    # ---------------------

    for i, model_name in enumerate(MODELS_TO_RUN):
        if i > 0:
            print("\n" + "="*80 + "\n") # 모델 분석 결과 사이에 구분선 추가
            
        try:
            model = get_analytic_model(model_name)
            run_risk_analysis(model=model, model_name=model_name)
        
        except Exception as e:
            print(f"\n'{model_name}' 모델 분석 중 오류 발생: {e}", file=sys.stderr)
            continue # 한 모델에서 오류가 나도 다음 모델 분석 계속

if __name__ == "__main__":
    main()