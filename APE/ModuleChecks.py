#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept  29 15:24:23 2023.

@author: Manu Goudar
"""

from pathlib import Path


def doesfileexist(filename):
    _p = Path(filename)
    flag = _p.is_file()
    if not flag:
        print("file is not present")
    return flag
