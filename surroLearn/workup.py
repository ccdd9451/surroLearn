#!/usr/bin/env python
# encoding: utf-8

from .recorder import Recorder

recorder = Recorder()


def find(method, var):
    method = eval(method)
    return recorder.find(method, var)


def argvar(var, arg, value):
    return arg, value, recorder.valueByArg(var, arg)


lines = []
