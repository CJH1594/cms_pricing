# -*- coding: utf-8 -*-
"""
시뮬레이션 유틸리티
===================

이 모듈은 가격결정에서 사용할 수 있는 시뮬레이션 함수들을 제공합니다.
현재는 G2++ 모델 경로 생성 함수만 래핑하여 제공합니다.
"""

from typing import Callable, Dict, Tuple
import numpy as np

from ..models.g2pp import simulate_g2_paths


def simulate_paths(params: Dict[str, float],
                   T_expiry: float,
                   num_paths: int,
                   num_steps: int,
                   phi_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """G2++ 경로 생성 함수에 대한 간단한 래퍼.
    가격결정 모듈에서 바로 사용할 수 있도록 이름을 추상화했습니다.
    """
    return simulate_g2_paths(params, T_expiry, num_paths, num_steps, phi_func)