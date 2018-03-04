#!/usr/bin/env python
# encoding: utf-8

from .recorder import Recorder
from pathlib import Path
from matplotlib import style
from matplotlib import pyplot as plt

style.use("ggplot")

_plots = []

recorder = Recorder()


class Plot(object):
    def __init__(self, name, save_dir):
        self._fig = plt.figure()
        self._ax = self._fig.subplots(1)
        self._name = name
        self._vars = name.split("|")
        self._dir = save_dir
        self._fname = str(Path(save_dir) / (name + ".svg"))

        _plots.append(self)

    def save(self):
        for var in self._vars:
            self.add_line(*recorder.serialize(var), label=var)
        self._ax.legend()
        self._fig.savefig(self._fname)
        self._fig.clear()
        self._ax = self._fig.subplots(1)

    def add_line(self, x, y, **kwargs):
        if not kwargs.get("label"):
            kwargs["label"] = self._name
        self._ax.plot(x, y, **kwargs)


def BroadcastSave():
    for p in _plots:
        p.save()


def PlotsClear():
    del _plots[:]
