#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_directories.py
====================

이 스크립트는 프로젝트의 기본 디렉터리를 생성합니다. 처음 실행 시 데이터
디렉터리(`data/`)가 없는 경우 자동으로 생성하며, 사용자 정의 작업을
추가할 수 있는 틀을 제공합니다.

사용 예::

    python3 setup_directories.py

"""

import os
import sys

# PYTHONPATH 설정: 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cms_pricing.config.settings import DATA_DIR


def main() -> None:
    # 데이터 디렉터리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"✓ 데이터 디렉터리 생성 또는 이미 존재: {DATA_DIR}")


if __name__ == "__main__":
    main()