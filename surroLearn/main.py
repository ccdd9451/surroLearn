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
    def __init__(self, filename, save_dir="./save", compatiable=True):
        self._filename = filename
        self._save_dir = save_dir
        if compatiable:
            from .data import compatible_load
            self._inputs, self._references = compatible_load(filename)

    def main(self):
        m = stack_max_out(1000, 10, 6)
        c = Constructor(m, self._inputs, self._references)
        g = c.training_bake()

        with tf.Session() as sess:
            e = Executor(sess, g, c.save_list, self._save_dir)
            for i in range(4):
                e.train()
                e.evaluate(*c.test_pipe())

    def lambda_inc(self, lr, steps):
        lr = eval(lr)
        steps = int(steps)

        from .ops import L2
        from .utils import tensor_linear_interval_range

        m = stack_max_out(1000, 10, 6)
        c = Constructor(m, self._inputs, self._references)

        def reg(w, *_):
            lt = tensor_linear_interval_range(*lr, steps)
            loss = L2(w, lt)
            loss = tf.identity(loss, name = "lambda_scale")
            return loss

        c.regularize_formulate(function=reg)

        g = c.training_bake()
        s = steps // 50
        t = Time(s)
        with tf.Session() as sess:
            e = Executor(sess, g, c.save_list, "")
            for i in range(s):
                e.train()
                e.evaluate(*c.test_pipe())
                print("Estimate time finishing", t.tick())
