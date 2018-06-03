#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import tensorflow as tf
import numpy as np

from collections import namedtuple
from pathlib import Path
from . import workup
from .constructor import Constructor
from .executor import Executor
from .maxout import stack_max_out
from .fullyConnected import stack_fc
from .utils import Time, export_graph
from .plot import Plot, BroadcastSave, PlotsClear
from .recorder import Recorder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Worklist = namedtuple("Worklist", ["inputs", "construct", "execute"])
lines = []


class Main(object):
    def __init__(self, save_dir=".", slots=100, batch_size=256):
        global path
        path = Path(save_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self.save_dir = str(path)
        self.slots = slots
        self.batch_size = batch_size

        self._worklist = Worklist._make([[], [], []])
        self._route = []
        self._data_size = None

        global lines
        del (lines[:])
        Recorder().clear()
        Recorder().path = str(path / "record.pkl")
        PlotsClear()

    def steps(self, steps):
        """ arg: steps  total step of training process """

        def w():
            self._steps = steps
            self._epoch_each = steps // self.slots

        self._worklist.inputs.append(w)

        return self

    def datasize(self, size):
        self._data_size = int(size)
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
        """ (1000, 10, 6) default maxout graph """

        def w():
            m = stack_max_out(1000, 10, 6)
            self._constructor = Constructor(
                m,
                self._inputs,
                self._references,
                shuffle_batch_size=self.batch_size,
            )

        self._worklist.construct.insert(0, w)

        return self

    def stack_maxout_conf(self, configs):
        """ (cell,unit,level) create a custom maxout graph """

        def w():
            m = stack_max_out(*configs)
            self._constructor = Constructor(m, self._inputs, self._references)

        self._worklist.construct.insert(0, w)

        return self

    def stack_fully_connected(self, activation_fn, configs):
        """ args: fn_name, (hidden_num,layer_num) fully connected graph """

        def w():
            m = stack_fc(activation_fn, *configs)
            self._constructor = Constructor(m, self._inputs, self._references)

        self._worklist.construct.insert(0, w)

        return self

    def export_graph(self):
        """ (optional) export whole training graph before training """

        def w():
            export_graph(self.save_dir)

        self._worklist.execute.append(w)

        return self

    def save_at_min(self, cls):
        """ save model when current tick is min of the history """

        def w():
            def r():
                if Recorder().ismin(cls):
                    self._executor.save_model()

            self._route.append(r)

        self._worklist.execute.append(w)

        return self

    def __smpl_train(self):
        """ basic training process with three pipes tested """

        def w():
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

            self._executor = Executor(
                tf.Session(config=config),
                self._constructor.graph,
                self._constructor.save_list,
                self.save_dir)
            self._route.insert(0, lambda: self._executor.evaluate("test"))
            self._route.insert(0,
                               lambda: self._executor.evaluate("cross_valid"))
            self._route.insert(0,
                               lambda: self._executor.train(self._epoch_each))

        self._worklist.execute.insert(0, w)

        return self

    def __go(self):
        for name, preparation in self._worklist._asdict().items():
            if len(preparation) == 0:
                raise
            for call in preparation:
                call()

        try:
            for i in range(self.slots):
                for call in self._route:
                    call()
        except KeyboardInterrupt as e:
            print(e)

    def varopt(self):
        """ args: batch_size using for batch gradient optimization """
        def w():
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

            self._executor = Executor(
                tf.Session(config=config),
                self._constructor.graph,
                self._constructor.save_list,
                self.save_dir)
            self._route.insert(0,
                               lambda: self._executor.input_opting(self._epoch_each))

        self._worklist.construct.append(
            lambda: self._constructor.opt_pipe_set(self.batch_size))
        self._worklist.construct.append(
            lambda: self._constructor.opt_bake())

        self._worklist.execute.insert(0, w)
        self.__go()

        return workupParser

    def predict(self):
        """ args: batch_size using for batch gradient optimization """
        def w():
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

            self._executor = Executor(
                tf.Session(config=config),
                self._constructor.graph,
                self._constructor.save_list,
                self.save_dir)

            def r():
                line = input("please input a new set of arguments")
                args = np.fromstring(line)[None, :]
                print(self._executor.predict(args))

            self._route.append(r)

        self._worklist.construct.append(
            lambda: self._constructor.training_bake())

        self._worklist.execute.insert(0, w)
        self.__go()

        return workupParser


    def train(self):
        """ last training command, workup arguments will follow """
        self._worklist.construct.append(
            lambda: self._constructor.training_bake())
        self.__smpl_train()
        self.__go()

        self._executor.save_model()
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

            def f():
                with open(str(path / "time.out"), "w") as f:
                    print(
                        "Esitmate time finishing",
                        t.tick(),
                        file=f,
                    )

            self._route.append(f)

        self._worklist.execute.append(w)
        return self

    def restore_from(self, load_dir):
        """ (optional arg: dir load model from certain dir) """

        def w():
            self._executor.load_model_from(load_dir)

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

    def classed_rmse(self, reg):
        """ (optional) arg: list of regularizer generate multiple rmses"""

        def w():
            from .formulations import classed_rmse
            self._constructor.rmse_loss_formulate(classed_rmse(reg))

        self._worklist.construct.append(w)

        return self

    _had_plot = False
    _had_tick = False

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
            self._route.append(lambda: BroadcastSave())
            self._had_plot = True
        if not self._had_tick:
            self._route.append(lambda: self._executor.tick())
            self._had_tick = True

        return self

    def observe_item(self, name):
        """ (optional) arg: var1|var2|... observe items but not plot """
        variables = name.split("|")
        for var in variables:
            if var not in "train cross_valid test".split():

                def w():
                    self._executor.add_tick(var)

                self._worklist.execute.append(w)

        if not self._had_tick:
            self._route.append(lambda: self._executor.tick())
            self._had_tick = True

        return self

    def plot_ctt(self):
        """ shortcut for plot_item "train|test|cross_valid" """
        self.plot_item("train|test|cross_valid")
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


def workupParser(pipe=None):
    if not pipe:
        if lines:
            lines.append("")
            filename = str(path / "workups")
            with open(filename, "w") as f:
                f.write("\n".join(lines))
        return

    result = []
    try:
        for method in pipe.split("|"):
            func, *argv = method.split(",")
            result = getattr(workup, func)(*(argv + result))
        lines.append(pipe + " => " + str(result))
    except IndexError as e:
        print("IndexError:", e)

    return workupParser


class TargetEmptyError(Exception):
    pass
