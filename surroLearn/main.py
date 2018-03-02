#!/usr/bin/env python
# encoding: utf-8

import os
import tensorflow as tf

from .constructor import Constructor
from .executor import Executor
from .maxout import stack_max_out
from .utils import Time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Main(object):
    def __init__(self):
        self.save_dir = ""

    def cfile(self, filename):

        from .data import compatible_load
        self._inputs, self._references = compatible_load(filename)

        return self

    def main(self):
        m = stack_max_out(1000, 10, 6)
        c = Constructor(m, self._inputs, self._references)
        g = c.training_bake()

        with tf.Session() as sess:
            e = Executor(sess, g, c.save_list, self.save_dir)
            for i in range(4):
                e.train()
                e.evaluate(*c.test_pipe())

    def lambda_inc(self, lr, steps):
        lr = eval(lr)
        steps = int(steps)

        from .formulations import linear_regularizer

        m = stack_max_out(1000, 10, 6)
        c = Constructor(m, self._inputs, self._references)

        c.regularize_formulate(function=linear_regularizer(*lr, steps))

        g = c.training_bake()
        s = steps // 50
        t = Time(s)
        with tf.Session() as sess:
            e = Executor(sess, g, c.save_list, "")
            for i in range(s):
                e.train()
                e.evaluate(*c.test_pipe())
                print("Estimate time finishing", t.tick())

    def ls(self):
        commands = [x for x in dir(self) if not x.startswith("_")]
        print("\t".join(commands))
