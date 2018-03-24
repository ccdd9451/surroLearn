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


# Rapid key for comparing records
def _recKey(x):
    return np.array(x[1]).sum()


class Recorder(defaultdict, metaclass=Singleton):
    timer = None
    path = "record.pkl"

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
            key=_recKey,  # summing for multi-var argument
        )

    def valueByArg(self, cls, arg):
        filted = filter(lambda x: x[0] == arg, self[cls])
        return list(filted)[0][1]

    def ismin(self, cls):
        query = self[cls]
        return query[-1] == min(query, key=_recKey)

    def dump(self):
        import pickle
        with open(self.path, "wb") as f:
            pickle.dump(self, f)
