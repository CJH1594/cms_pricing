# -*- coding: utf-8 -*-
"""
`pricing` 모듈은 상품 가격결정 기능을 제공합니다. 현재는 CMS 스프레드
디지털 노트에 대한 가격결정 함수가 구현되어 있습니다.
"""

from .simulation import simulate_paths  # 재사용을 위한 래퍼
from .cms_digital import price_digital_cms_spread

__all__ = [
    "simulate_paths",
    "price_digital_cms_spread",
]