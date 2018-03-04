#!/usr/bin/env python
# encoding: utf-8

from collections import defaultdict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Recorder(defaultdict, metaclass=Singleton):
    timer = None

    def __init__(self):
        super(Recorder, self).__init__(list)

    def record(self, cls, value):
        if not self.timer:
            raise RuntimeError("timer not being set correctly")
        time = self.timer()
        self[cls].append([time, value])

    def serialize(self, cls):
        return list(map(list, zip(*self[cls])))

    def find(self, func, cls):
        return func(
            self[cls],
            key=lambda x: x[1],
        )

    def valueByArg(self, cls, arg):
        filted = filter(lambda x: x[0] == arg, self[cls])
        return list(filted)[0][1]
