#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from .recorder import Recorder

np.set_printoptions(precision=4, suppress=True)
recorder = Recorder()

def find(method, var):
    method = eval(method)
    return recorder.find(method, var)


def argvar(var, arg, value):
    return arg, str(value), str(recorder.valueByArg(var, arg))


def dump():
    recorder.dump()
    return recorder.path

lines = []
