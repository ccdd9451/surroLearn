#!/usr/bin/env python
# encoding: utf-8

import sys
import fire
import pickle
import unittest
import numpy as np
import surroLearn as sl
import tensorflow as tf

from pathlib import Path
from tempfile import TemporaryDirectory
from mock import patch


class MainTest(tf.test.TestCase):
    def _fakeDataTest(self, command):
        with TemporaryDirectory() as tdname:
            tempdata = Path(tdname) / "data"
            tempdata = str(tempdata)
            inputs = np.random.randn(500, 10)
            references = np.random.randn(500)
            with open(tempdata, "wb") as f:
                pickle.dump({
                    "X": inputs,
                    "Y": references,
                }, f)
            testargs = command.format(tempdata)
            with patch.object(sys, 'argv', testargs.split()):
                fire.Fire(sl.main.Main)

    #@unittest.skip("temp")
    def test_Plotting_Multis(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/pm "
                           "--slots=5 cfile {} steps 1000 "
                           "stack_maxout smpl_train "
                           "plot_item train|cross_valid|test "
                           "plot_item lambda_scale "
                           "lambda_inc (0,0.1) train")

    def test_MainStream_l2_inc(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/li "
                           "--slots=10 cfile {} steps 1000 "
                           "stack_maxout smpl_train "
                           "plot_item train|cross_valid|test "
                           "lambda_inc (0,0.01) train")

    def test_MainStream_l2_static(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/ls "
                           "--slots=5 cfile {} steps 1000 "
                           "stack_maxout smpl_train "
                           "plot_item train|cross_valid|test "
                           "lambda_static 0.1 train")

    def test_L2_graph_export(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/lge "
                           "--slots=5 cfile {} steps 1000 "
                           "stack_maxout smpl_train "
                           "lambda_inc (0,0.01) export_graph train")

    def test_Workup_Performance(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/wp "
                           "--slots=5 cfile {} steps 1000 "
                           "stack_maxout smpl_train "
                           "plot_item train|cross_valid|test "
                           "plot_item lambda_scale "
                           "lambda_inc (0,0.001) export_graph train "
                           "find,min,cross_valid|argvar,train "
                           "find,min,cross_valid|argvar,test")

    def test_CLI_ls(self):
        testargs = ["learn", "ls"]
        with patch.object(sys, 'argv', testargs):
            fire.Fire(sl.main.Main)
