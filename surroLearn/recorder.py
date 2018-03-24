#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from collections import defaultdict, Iterable


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
        if not self[cls]:
            return None
        time, data = list(zip(*self[cls]))
        time = np.array(time)
        data = np.array(data)

        return [time, data]

    def find(self, func, cls):
        return func(
            self[cls],
            key=lambda x: x[1],
        )

    def valueByArg(self, cls, arg):
        filted = filter(lambda x: x[0] == arg, self[cls])
        return list(filted)[0][1]

    def ismin(self, cls):
        query = self[cls]
        return query[-1] == min(query)
