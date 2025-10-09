# cms_pricing/src/models/analytic.py

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import interp1d
import json
import os

# --- 할인계수 곡선 로더 및 보간 함수 ---
def _get_discount_interpolator():
    """
    market_data.json에서 P(0,T) 데이터를 로드하고,
    로그 선형 보간(log-linear interpolation)을 수행하는 함수를 반환합니다.
    """
    try:
        current_dir = os.path.dirname(__file__)
        data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'market_data.json'))
        with open(data_path, 'r') as f:
            market_data = json.load(f)
        p_year_data = market_data['P_year']
    except Exception:
        # 파일 로드 실패 시, 비상용 데이터 사용
        p_year_data = {"1": 0.96, "10": 0.7, "30": 0.3}

    maturities = np.array([0] + [float(t) for t in p_year_data.keys()])
    dfs = np.array([1.0] + list(p_year_data.values()))

    sort_indices = np.argsort(maturities)
    maturities, dfs = maturities[sort_indices], dfs[sort_indices]

    log_dfs = np.log(dfs)
    log_linear_interpolator = interp1d(maturities, log_dfs, kind='linear', fill_value="extrapolate")

    def interpolator(t: float) -> float:
        return np.exp(log_linear_interpolator(t))

    return interpolator

_get_p_t = _get_discount_interpolator()


class AnalyticModel(ABC):
    """
    GBM, Bachelier 등 분석적 해법이 존재하는 모델의 공통 인터페이스.
    모든 가격 및 리스크는 현재가치(PV) 기준입니다.
    """
    def _get_discount_factor(self, T: float) -> float:
        """내부 보간 함수를 사용해 할인계수를 가져옵니다."""
        return _get_p_t(T)

    @abstractmethod
    def price_digital(self, S: float, K: float, T: float, vol: float) -> float:
        """디지털 옵션의 현재가치(PV)를 계산한다."""
        pass

    @abstractmethod
    def delta_digital(self, S: float, K: float, T: float, vol: float) -> float:
        """디지털 옵션의 현재가치 기준 델타를 계산한다."""
        pass
        
    @abstractmethod
    def gamma_digital(self, S: float, K: float, T: float, vol: float) -> float:
        """디지털 옵션의 현재가치 기준 감마를 계산한다."""
        pass

    @abstractmethod
    def implied_vol(self, S: float, K: float, T: float, target_pv: float) -> tuple[float, bool]:
        """현재가치로부터 디지털 옵션의 내재변동성을 역산한다."""
        pass

class GBMModel(AnalyticModel):
    """GBM (Black-Scholes) 모델 구현 (PV 기준)."""
    
    def price_digital(self, S: float, K: float, T: float, vol: float) -> float:
        P_T = self._get_discount_factor(T)
        if vol <= 1e-9 or T <= 0 or S <= 0 or K <= 0:
            return P_T * (1.0 if S > K else 0.0)
        d2 = (np.log(S / K) - 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        return P_T * norm.cdf(d2)

    def delta_digital(self, S: float, K: float, T: float, vol: float) -> float:
        P_T = self._get_discount_factor(T)
        if vol <= 1e-9 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        d2 = (np.log(S / K) - 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        return P_T * norm.pdf(d2) / (S * vol * np.sqrt(T))
        
    def gamma_digital(self, S: float, K: float, T: float, vol: float) -> float:
        P_T = self._get_discount_factor(T)
        if vol <= 1e-9 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        
        sigma_sqrt_T = vol * np.sqrt(T)
        d1 = (np.log(S / K) + 0.5 * vol**2 * T) / sigma_sqrt_T
        phi_d2 = norm.pdf(d1 - sigma_sqrt_T)
        
        return P_T * -phi_d2 * d1 / (S**2 * vol**2 * T)

    def implied_vol(self, S: float, K: float, T: float, target_pv: float) -> tuple[float, bool]:
        P_T = self._get_discount_factor(T)
        if P_T < 1e-9: return 0.0, False
        
        target_prob = target_pv / P_T
        # P_T 없이 확률만 계산하는 함수를 만듭니다.
        price_fn_prob = lambda v: self.price_digital(S, K, T, v) / self._get_discount_factor(T)
        objective_prob = lambda vol: target_prob - price_fn_prob(vol)

        if np.isclose(S, K):
            return (0.20, True) if np.isclose(target_prob, 0.5) else (0.0, False)
        try:
            vol_peak = np.sqrt(max(0.0, 2.0 * abs(np.log(S / K)) / T))
            vol_min, vol_max = 1e-8, max(vol_peak, 1e-6)
            f_min, f_max = objective_prob(vol_min), objective_prob(vol_max)
            if f_min * f_max >= 0: return 0.0, False
            vol = brentq(objective_prob, vol_min, vol_max, xtol=1e-12, rtol=1e-12)
            return vol, True
        except (ValueError, RuntimeError):
            return 0.0, False

class BachelierModel(AnalyticModel):
    """Bachelier (Normal) 모델 구현 (PV 기준)."""

    def price_digital(self, S: float, K: float, T: float, vol: float) -> float:
        P_T = self._get_discount_factor(T)
        if vol <= 1e-9 or T <= 0:
            return P_T * (1.0 if S > K else 0.0)
        d = (S - K) / (vol * np.sqrt(T))
        return P_T * norm.cdf(d)

    def delta_digital(self, S: float, K: float, T: float, vol: float) -> float:
        P_T = self._get_discount_factor(T)
        if vol <= 1e-9 or T <= 0:
            return 0.0
        d = (S - K) / (vol * np.sqrt(T))
        return P_T * norm.pdf(d) / (vol * np.sqrt(T))
        
    def gamma_digital(self, S: float, K: float, T: float, vol: float) -> float:
        P_T = self._get_discount_factor(T)
        if vol <= 1e-9 or T <= 0:
            return 0.0
        d = (S - K) / (vol * np.sqrt(T))
        return P_T * -d * norm.pdf(d) / (vol**2 * T)

    def implied_vol(self, S: float, K: float, T: float, target_pv: float) -> tuple[float, bool]:
        P_T = self._get_discount_factor(T)
        if P_T < 1e-9: return 0.0, False

        target_prob = target_pv / P_T
        price_fn_prob = lambda v: self.price_digital(S, K, T, v) / self._get_discount_factor(T)
        objective_prob = lambda vol: target_prob - price_fn_prob(vol)
        
        if np.isclose(S, K):
            return (0.005, True) if np.isclose(target_prob, 0.5) else (0.0, False)
        try:
            vol_min, vol_max = 1e-8, 5.0
            f_min, f_max = objective_prob(vol_min), objective_prob(vol_max)
            if f_min * f_max >= 0: return 0.0, False
            vol = brentq(objective_prob, vol_min, vol_max, xtol=1e-12, rtol=1e-12)
            return vol, True
        except (ValueError, RuntimeError):
            return 0.0, False

def get_analytic_model(model_name: str) -> AnalyticModel:
    if model_name.lower() == 'bachelier':
        return BachelierModel()
    elif model_name.lower() == 'gbm':
        return GBMModel()
    else:
        raise ValueError(f"지원되지 않는 분석 모델입니다: {model_name}")