# cms_pricing/src/models/analytic.py

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import brentq
from scipy.stats import norm

class AnalyticModel(ABC):
    """
    GBM, Bachelier 등 분석적 해법이 존재하는 모델의 공통 인터페이스.
    """
    @abstractmethod
    def price_digital(self, S: float, K: float, T: float, vol: float) -> float:
        """디지털 옵션 가격을 계산한다."""
        pass

    @abstractmethod
    def delta_digital(self, S: float, K: float, T: float, vol: float) -> float:
        """디지털 옵션 델타를 계산한다."""
        pass
        
    @abstractmethod
    def gamma_digital(self, S: float, K: float, T: float, vol: float) -> float:
        """디지털 옵션 감마를 계산한다."""
        pass

    @abstractmethod
    def implied_vol(self, S: float, K: float, T: float, target_price: float) -> tuple[float, bool]:
        """디지털 옵션 내재변동성을 역산한다."""
        pass

class GBMModel(AnalyticModel):
    """GBM (Black-Scholes) 모델 구현."""
    
    def price_digital(self, S: float, K: float, T: float, vol: float) -> float:
        if vol <= 1e-9 or T <= 0 or S <= 0 or K <= 0:
            return 1.0 if S > K else 0.0
        d2 = (np.log(S / K) - 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        return norm.cdf(d2)

    def delta_digital(self, S: float, K: float, T: float, vol: float) -> float:
        if vol <= 1e-9 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        d2 = (np.log(S / K) - 0.5 * vol**2 * T) / (vol * np.sqrt(T)) # Use d2 here
        return norm.pdf(d2) / (S * vol * np.sqrt(T))
        
    def gamma_digital(self, S: float, K: float, T: float, vol: float) -> float:
        if vol <= 1e-9 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (np.log(S / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        phi_d1 = norm.pdf(d1)
        # Gamma formula for digital options derived from d(Delta)/dS
        return -phi_d1 * (d1 + vol * np.sqrt(T)) / (S**2 * (vol * np.sqrt(T))**2)

    def implied_vol(self, S: float, K: float, T: float, target_price: float) -> tuple[float, bool]:
        objective = lambda vol: target_price - self.price_digital(S, K, T, vol)
        if np.isclose(S, K):
            return (0.20, True) if np.isclose(target_price, 0.5) else (0.0, False)
        try:
            # This logic finds the smaller of two possible roots for GBM
            vol_peak = np.sqrt(max(0.0, 2.0 * abs(np.log(S / K)) / T))
            vol_min, vol_max = 1e-8, max(vol_peak, 1e-6)
            f_min, f_max = objective(vol_min), objective(vol_max)
            if f_min * f_max >= 0: return 0.0, False
            vol = brentq(objective, vol_min, vol_max, xtol=1e-12, rtol=1e-12)
            return vol, True
        except (ValueError, RuntimeError):
            return 0.0, False

class BachelierModel(AnalyticModel):
    """Bachelier (Normal) 모델 구현."""

    def price_digital(self, S: float, K: float, T: float, vol: float) -> float:
        if vol <= 1e-9 or T <= 0:
            return 1.0 if S > K else 0.0
        d = (S - K) / (vol * np.sqrt(T))
        return norm.cdf(d)

    def delta_digital(self, S: float, K: float, T: float, vol: float) -> float:
        if vol <= 1e-9 or T <= 0:
            return 0.0
        d = (S - K) / (vol * np.sqrt(T))
        return norm.pdf(d) / (vol * np.sqrt(T))
        
    def gamma_digital(self, S: float, K: float, T: float, vol: float) -> float:
        if vol <= 1e-9 or T <= 0:
            return 0.0
        d = (S - K) / (vol * np.sqrt(T))
        return -d * norm.pdf(d) / (vol**2 * T)

    def implied_vol(self, S: float, K: float, T: float, target_price: float) -> tuple[float, bool]:
        if np.isclose(S, K):
            return (0.005, True) if np.isclose(target_price, 0.5) else (0.0, False)
        objective = lambda vol: target_price - self.price_digital(S, K, T, vol)
        try:
            # Bachelier is monotonic, so search in a wide bracket
            vol_min, vol_max = 1e-8, 5.0
            f_min, f_max = objective(vol_min), objective(vol_max)
            if f_min * f_max >= 0: return 0.0, False
            vol = brentq(objective, vol_min, vol_max, xtol=1e-12, rtol=1e-12)
            return vol, True
        except (ValueError, RuntimeError):
            return 0.0, False

def get_analytic_model(model_name: str) -> AnalyticModel:
    """모델 이름에 따라 적절한 분석 모델 객체를 반환하는 팩토리 함수."""
    if model_name.lower() == 'bachelier':
        return BachelierModel()
    elif model_name.lower() == 'gbm':
        return GBMModel()
    else:
        raise ValueError(f"지원되지 않는 분석 모델입니다: {model_name}")