#!/usr/bin/env python
# encoding: utf-8

import os
import tensorflow as tf

from collections import namedtuple
from pathlib import Path
from . import workup
from .constructor import Constructor
from .executor import Executor
from .maxout import stack_max_out
from .utils import Time, export_graph
from .plot import Plot, BroadcastSave, PlotsClear
from .recorder import Recorder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Worklist = namedtuple("Worklist", ["inputs", "construct", "execute"])


class Main(object):
    def __init__(self, save_dir=".", slots=200):
        global path
        path = Path(save_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.save_dir = str(path)
        self.slots = slots

        self._worklist = Worklist._make([[], [], []])
        self._route = []
        self._data_size = None

        Recorder().clear()
        PlotsClear()

    def steps(self, steps):
        """ arg: steps  total step of training process """

        def w():
            self._steps = steps
            self._epoch_each = steps // self.slots

        self._worklist.inputs.append(w)

        return self

    def datasize(self, size):
        self._data_size = size
        return self

    def cfile(self, filename):
        """ arg: filename  dataset from compatible_load """

        def w():
            from .data import compatible_load
            inp, ref = compatible_load(filename)
            self._inputs = inp[:self._data_size]
            self._references = ref[:self._data_size]

        self._worklist.inputs.append(w)

        return self

    def stack_maxout(self):
        """ default maxout graph """

        def w():
            m = stack_max_out(1000, 10, 6)
            self._constructor = Constructor(m, self._inputs, self._references)

        self._worklist.construct.insert(0, w)

        return self

    def stack_maxout_conf(self, configs):
        """ default maxout graph """

        def w():
            m = stack_max_out(*configs)
            self._constructor = Constructor(m, self._inputs, self._references)

        self._worklist.construct.insert(0, w)

        return self

    def export_graph(self):
        """ (optional) export whole training graph before training """

        def w():
            export_graph(self.save_dir)

        self._worklist.execute.append(w)

        return self

    def smpl_train(self):
        """ basic training process with three pipes tested """

        def w():

            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

            self._executor = Executor(
                tf.Session(config=config),
                self._constructor.training_bake(),
                self._constructor.save_list,
                self.save_dir)
            self._route.insert(0, lambda: self._executor.evaluate("test"))
            self._route.insert(0,
                               lambda: self._executor.evaluate("cross_valid"))
            self._route.insert(0,
                               lambda: self._executor.train(self._epoch_each))

        self._worklist.execute.insert(0, w)

        return self

    def train(self):
        """ last training command, workup arguments will follow """
        for name, preparation in self._worklist._asdict().items():
            if len(preparation) == 0:
                raise
            for call in preparation:
                call()

        for i in range(self.slots):
            for call in self._route:
                call()
        return workupParser

    def ptrain(self):
        """ last training command, workup arguments will follow """
        for name, preparation in self._worklist._asdict().items():
            if len(preparation) == 0:
                raise
            for call in preparation:
                call()

        import profile
        profile.runctx("""
for i in range(self.slots):
    for call in self._route:
        call()
        """, globals(), locals())
        return workupParser

    def timeit(self):
        """ (optional) add time estimation to the training process"""

        def w():
            t = Time(self.slots)
            self._route.append(
                lambda: print("Esitmate time finishing", t.tick()))

        self._worklist.execute.append(w)
        return self

    def lambda_inc(self, lr):
        """ (optional) arg: [lrL, lrR] linear increasing lambda """

        def w():
            from .formulations import linear_regularizer
            self._constructor.regularize_formulate(
                linear_regularizer(*lr, self._steps))

        self._worklist.construct.append(w)

        return self

    def lambda_static(self, lr):
        """ (optional) arg: [lrL, lrR] linear increasing lambda """

        def w():
            self._constructor.regularize_formulate(lr)

        self._worklist.construct.append(w)

        return self

    _had_plot = False

    def plot_item(self, name):
        """ (optional) arg: var1|var2|... plotting images """
        variables = name.split("|")
        Plot(name, self.save_dir)
        for var in variables:
            if var not in "train cross_valid test".split():

                def w():
                    self._executor.add_tick(var)

                self._worklist.execute.append(w)

        if not self._had_plot:
            self._route.append(lambda: self._executor.tick())
            self._route.append(lambda: BroadcastSave())
        return self

    def ls(self):
        """ this help command """
        lines = []
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            else:
                item = getattr(self, attr)
                if callable(item):
                    lines.append("  {:17}{}".format(attr, item.__doc__))
                else:
                    lines.insert(0, "  --" + attr)

        print("\n".join(lines))


lines = []


def workupParser(pipe=None):
    if not pipe:
        if lines:
            filename = str(path / "workups")
            with open(filename, "w") as f:
                f.write("\n".join(lines))
        return

    result = []
    for method in pipe.split("|"):
        func, *argv = method.split(",")
        result = getattr(workup, func)(*(argv + result))
    lines.append(pipe + " => " + str(result))

    return workupParser


class TargetEmptyError(Exception):
    pass
