#!/usr/bin/env python
# encoding: utf-8

import os
import tensorflow as tf

from .constructor import Constructor
from .executor import Executor
from .maxout import stack_max_out
from .data import unittest_sample


def main(compatible_load_filename=None, save_dir="./save"):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if compatible_load_filename:
        from .data import compatible_load
        inputs, references = compatible_load(compatible_load_filename)
    else:
        inputs, references = unittest_sample()

    m = stack_max_out(1000, 10, 6)
    c = Constructor(m, inputs, references)
    g = c.training_bake()

    with tf.Session() as sess:
        e = Executor(sess, g, c.save_list, save_dir)
        for i in range(4):
            e.train()
            e.evaluate(*c.test_pipe())
