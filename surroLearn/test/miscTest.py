#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import tensorflow as tf
import surroLearn as sl

sys.stdout = open(os.devnull, 'w')


class MiscTest(tf.test.TestCase):
    def test_expo_decay_tensor(self):
        g = tf.Graph()
        with self.test_session(graph=g) as _, g.as_default():
            g = tf.get_variable("global_step", dtype=tf.int32, initializer=0)
            r = sl.utils.tensor_geo_interval_range(1, 0.001, 100)
            tf.global_variables_initializer().run()
            self.assertAlmostEqual(1, r.eval())
            g.assign(100).eval()
            self.assertAlmostEqual(0.001, r.eval())
