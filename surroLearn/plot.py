#!/usr/bin/env python
# encoding: utf-8

from pathlib import Path
from matplotlib import style
from matplotlib import pyplot as plt

style.use("ggplot")

_plots = []


class Plot(object):
    def __init__(self, name, save_dir):
        self._fig = plt.figure()
        self._ax = plt.subplot(111)
        self._name = name
        self._dir = save_dir
        self._fname = str(Path(save_dir) / (name + ".svg"))

        _plots.append(self)

    def save(self):
        self._ax.legend()
        self._fig.savefig(self._fname)
        self._fig.clear()

    def add_line(self, x, y, **kwargs):
        self._ax.plot(x, y, **kwargs)


def BroadcastSave():
    for p in _plots:
        p.Save()
