#!/usr/bin/env python
# encoding: utf-8

import os
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

sys.stdout = open(os.devnull, 'w')


class MainTest(tf.test.TestCase):
    def _fakeDataTest(self, command, size=500):
        with TemporaryDirectory() as tdname:
            tempdata = Path(tdname) / "data"
            tempdata = str(tempdata)
            inputs = np.random.randn(size, 10)
            references = np.random.randn(size)
            with open(tempdata, "wb") as f:
                pickle.dump({
                    "X": inputs,
                    "Y": references,
                }, f)
            testargs = command.format(tempdata)
            with patch.object(sys, 'argv', testargs.split()):
                fire.Fire(sl.main.Main)

    def _fakeDataTestMulticlass(self, command, size=500):
        with TemporaryDirectory() as tdname:
            tempdata = Path(tdname) / "data"
            tempdata = str(tempdata)
            inputs = np.random.randn(size, 10)
            references = np.random.randn(size, 3)
            with open(tempdata, "wb") as f:
                pickle.dump({
                    "X": inputs,
                    "Y": references,
                }, f)
            testargs = command.format(tempdata)
            with patch.object(sys, 'argv', testargs.split()):
                fire.Fire(sl.main.Main)

    @unittest.skip("same func in test_Workup_Performance")
    def test_Plotting_Multis(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/pm "
                           "--slots=5 cfile {} steps 10 "
                           "stack_maxout "
                           "plot_item train|cross_valid|test "
                           "plot_item lambda_scale "
                           "lambda_inc (0,0.1) train")

    def test_multiClass(self):
        self._fakeDataTestMulticlass("learn "
                                     "--save_dir=.pytest_cache/mc "
                                     "--slots=5 cfile {} steps 10 "
                                     "stack_maxout classed_rmse (1,1,1) "
                                     "plot_ctt plot_item lambda_scale "
                                     "lambda_inc (0,0.1) train "
                                     "find,min,cross_valid|argvar,train "
                                     )

    @unittest.skip("same func in test_Workup_Performance")
    def test_MainStream_l2_inc(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/li "
                           "--slots=10 cfile {} steps 10 "
                           "stack_maxout "
                           "plot_item train|cross_valid|test "
                           "lambda_inc (0,0.01) train")

    @unittest.skip("Unknown error, repair it later")
    def test_Fully_Connected(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/fc "
                           "--slots=10 cfile {} steps 10 "
                           "stack_fully_connected relu (100,2) "
                           "plot_item train|cross_valid|test "
                           "train")

    def test_MainStream_l2_static(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/ls "
                           "--slots=5 cfile {} steps 10 "
                           "stack_maxout "
                           "plot_item train|cross_valid|test "
                           "lambda_static 0.1 train")

    def test_Simplest_Timeit(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/st "
                           "--slots=5 cfile {} steps 10 "
                           "stack_maxout "
                           "timeit train")

    def test_Simplest_Datasize(self):
        self._fakeDataTest(
            "learn --save_dir=.pytest_cache/sd "
            "--slots=5 cfile {} steps 10 "
            "stack_maxout datasize 500 "
            "timeit train",
            size=5000)

    def test_L2_graph_export(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/lge "
                           "--slots=5 cfile {} steps 10 "
                           "stack_maxout "
                           "lambda_inc (0,0.01) export_graph train")

    def test_Workup_Performance(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/wp "
                           "--slots=5 cfile {} steps 10 "
                           "stack_maxout "
                           "plot_item train|cross_valid|test "
                           "plot_item lambda_scale "
                           "lambda_inc (0,0.1) train "
                           "find,min,cross_valid|argvar,train "
                           "find,min,cross_valid|argvar,test")

    def test_dump_test(self):
        self._fakeDataTest("learn --save_dir=.pytest_cache/dp "
                           "--slots=5 cfile {} steps 10 "
                           "stack_maxout "
                           "plot_item train|cross_valid|test "
                           "plot_item lambda_scale "
                           "lambda_inc (0,0.1) train "
                           "dump")

    def test_CLI_ls(self):
        testargs = ["learn", "ls"]
        with patch.object(sys, 'argv', testargs):
            fire.Fire(sl.main.Main)
