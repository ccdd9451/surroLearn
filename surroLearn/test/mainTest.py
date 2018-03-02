#!/usr/bin/env python
# encoding: utf-8

import pickle
import numpy as np
import surroLearn as sl
import tensorflow as tf

from pathlib import Path
from tempfile import TemporaryDirectory


class MainTest(tf.test.TestCase):
    def test_MainStream_Simplest_w_Cmpfile(self):
        with TemporaryDirectory() as tdname:
            tempdata = Path(tdname) / "data"
            inputs = np.random.randn(100, 10)
            references = np.random.randn(100)
            with open(tempdata, "wb") as f:
                pickle.dump({
                    "X": inputs,
                    "Y": references,
                }, f)

            sl.main.main(str(tempdata))


    def test_MainStream_l2_exp_inc(self):
        with TemporaryDirectory() as tdname:
            tempdata = Path(tdname) / "data"
            inputs = np.random.randn(100, 10)
            references = np.random.randn(100)
            with open(tempdata, "wb") as f:
                pickle.dump({
                    "X": inputs,
                    "Y": references,
                }, f)

            sl.main.lambda_inc(str(tempdata), "[10**-8, 1]", 500)
