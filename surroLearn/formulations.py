#!/usr/bin/env python
# encoding: utf-8

from .ops import L2

def l2_lambda(scale):
    def reg(w, _=None):
        nonlocal scale
        return sum(L2(w, scale))
