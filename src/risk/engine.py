# cms_pricing/src/risk/engine.py

import numpy as np
from cms_pricing.src.models.analytic import AnalyticModel
from cms_pricing.src.pricing.utils import load_pricing_results
from cms_pricing.config.settings import (
    RISK_SHOCK_AMOUNT,
)

def run_risk_analysis(model: AnalyticModel, model_name: str) -> None:
    """
    주어진 가격결정 모델을 사용하여 전체 리스크 분석을 수행하고 결과를 출력한다.
    """
    print(f"===== {model_name.upper()} 모델을 사용한 리스크 분석 =====")
    
    # 1. 데이터 로드
    try:
        pricing_results = load_pricing_results()
        prod = pricing_results["product"]
        note_value = pricing_results["price"]
    except (FileNotFoundError, RuntimeError, ValueError) as e: # Add ValueError here
        raise RuntimeError("가격 계산 결과 파일(pricing_results.json)이 없습니다. 03_price_product.py를 먼저 실행하세요.") from e

    # 2. 내재 변동성 계산
    S = prod['spread_asset_price']
    K = prod['strike']
    T = prod['expiry']
    
    target_price = float(np.clip(note_value / (prod['notional'] * prod['coupon']), 1e-9, 1.0 - 1e-9))
    
    implied_vol, converged = model.implied_vol(S, K, T, target_price)
    if not converged:
        print("\n경고: 내재 변동성 수렴 실패. 분석의 정확도가 낮을 수 있습니다.")

    # 3. 델타 및 감마 계산
    delta_underlying = model.delta_digital(S, K, T, implied_vol)
    gamma_underlying = model.gamma_digital(S, K, T, implied_vol)
    
    delta_long, delta_short = delta_underlying, -delta_underlying
    
    # 4. 초기 상태 출력
    print("\n로드된 상품 스펙:")
    print(f"  만기 (T): {T}년, Strike (K): {K:.6f}")
    print(f"  Spread (S): {S:.6f}")
    print(f"  노트 가치: {note_value:.6f}")
    print(f"\n계산된 리스크:")
    print(f"  내재 변동성 (σ): {implied_vol:.6f}")
    print(f"  델타 (Long/Short): {delta_long:.4f} / {delta_short:.4f}")
    print(f"  감마: {gamma_underlying:.4f}")

    # 5. 쇼크 시뮬레이션
    shock = RISK_SHOCK_AMOUNT # Use RISK_SHOCK_AMOUNT from settings
    S_new_spread = S + shock # 스프레드에 직접 쇼크

    price_change_approx = delta_underlying * shock + 0.5 * gamma_underlying * shock**2
    note_value_new_approx = note_value + price_change_approx

    hedge_pnl = delta_long * shock + delta_short * 0 # Short leg은 고정되었다고 가정
    portfolio_pnl = (note_value_new_approx - note_value) - hedge_pnl
    
    print(f"\n스프레드 {shock*1e4:.0f}bp 상승 시 (델타-감마 근사):")
    print(f"  예상 노트 가치 변화: {price_change_approx:+.6f}")
    print(f"  헤지 포트폴리오 P&L (손익): {portfolio_pnl:+.6f}")