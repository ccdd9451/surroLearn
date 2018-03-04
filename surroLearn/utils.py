#!/usr/bin/env python
# encoding: utf-8

import time
import tensorflow as tf

from datetime import datetime


def export_graph(path):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(str(path), sess.graph)
        writer.close()


class Time(object):
    def __init__(self, tick_amount):
        self._count = 0
        self._tick_amount = tick_amount
        self._start_time = time.time()

    def now(self):
        return datetime.now()

    def tick(self):
        self._count += 1
        self._latest_ticking = time.time()
        return self.predicting_time()

    def percentage(self):
        return self._count / self._tick_amount

    def predicting_time(self):
        return datetime.fromtimestamp(self._start_time + (
            self._latest_ticking - self._start_time) / self.percentage())


def tensor_geo_interval_range(begin, end, step_amount):
    with tf.variable_scope("", reuse=True):
        global_step = tf.get_variable(name="global_step", dtype=tf.int32)
    return tf.train.exponential_decay(
        float(begin), global_step, step_amount, end / begin)


def tensor_linear_interval_range(begin, end, step_amount):
    with tf.variable_scope("", reuse=True):
        global_step = tf.get_variable(name="global_step", dtype=tf.int32)
        global_step = tf.cast(global_step, tf.float32)
    return (global_step / step_amount) * (end - begin) + begin


def nilfunc(*argv):
    pass
