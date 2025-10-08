# -*- coding: utf-8 -*-
"""
`models` 패키지는 금리 모형(G2++ 등)의 정의와 파라미터 보정 로직을 포함합니다.

* ``g2pp``: G2++ 모형 관련 함수 (φ(t), 시뮬레이션 등)
* ``calibration``: 스왑션 시장 데이터를 사용한 G2++ 파라미터 보정
"""

from .analytic import (
    get_analytic_model,
    AnalyticModel,
    GBMModel,
    BachelierModel,
)

from .g2pp import (
    make_lnP_spline,
    phi_g2pp_factory,
    simulate_g2_paths,
    calculate_forward_swap_rate,
    calculate_V,
    B,
)
from .calibration import (
    calibrate_g2pp,
    load_calibrated_params,
)

__all__ = [
    "get_analytic_model",
    "AnalyticModel",
    "GBMModel",
    "BachelierModel",
    "make_lnP_spline",
    "phi_g2pp_factory",
    "simulate_g2_paths",
    "calculate_forward_swap_rate",
    "calculate_V",
    "B",
    "black_swaption_price",
    "calculate_forward_swap_rate_market",
    "build_market_prices_from_vol_surface",
    "calibrate_g2pp",
    "load_calibrated_params",
]