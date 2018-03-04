#!/usr/bin/env python
# encoding: utf-8

import os
import tensorflow as tf

from collections import namedtuple
from .constructor import Constructor
from .executor import Executor
from .maxout import stack_max_out
from .utils import Time
from .plot import Plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Worklist = namedtuple("Worklist", ["inputs", "construct", "execute"])


class Main(object):
    def __init__(self):
        self.save_dir = "."
        self.train_batch = 50

        self._worklist = Worklist([[]] * 3)
        self._route = []

    def steps(self, steps):
        def w():
            self._steps = steps
            self._train_times = steps // self.train_batch

        self._worklist.inputs.append(w)

        return self

    def cfile(self, filename):
        def w():
            from .data import compatible_load
            self._inputs, self._references = compatible_load(filename)

        self._worklist.inputs.append(w)

        return self

    def stack_max_out(self):
        def w():
            m = stack_max_out(1000, 10, 6)
            self._c = Constructor(m, self._inputs, self._references)

        self._worklist.construct.insert(0, w)

        return self

    def smpl_train(self):
        def w():
            self._executor = Executor(tf.Session(),
                                      self._constructor.training_bake(),
                                      self._constructor.save_list,
                                      self.save_dir)
            self._route.insert(0, lambda: self._executor.train(self.train_batch))
            self._route.insert(1, lambda: self._executor.evaluate(*self._c.test_pipe()))

        self._worklist.execute.insert(0, w)

        return self

    def train(self):
        for preparation in self._worklist:
            for call in preparation:
                call()

        for i in range(self._steps // self.train_batch):
            for call in self._route:
                call()

    def timeit(self):
        t = Time(self._train_times)
        self._route.append(lambda: print("Esitmate time finishing", t.tick()))
        return self

    def lambda_inc(self, lr):
        lr = eval(lr)

        def w():
            from .formulations import linear_regularizer
            self._c.regularize_formulate(
                function=linear_regularizer(*lr, self._steps))

        self._worklist.construct.append(w)

        return self

    def plotitem(self, name):
        variables = name.split("|")

    def ls(self):
        commands = [x for x in dir(self) if not x.startswith("_")]
        print("\t".join(commands))
