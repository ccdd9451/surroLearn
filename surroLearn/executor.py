#!/usr/bin/env python
# encoding: utf-8

import os
import tensorflow as tf

from datetime import datetime


class Executor(object):
    def __init__(self, sess, graph, save_list, save_dir, evaluate_only=False):
        """

        With a finalized graph and a session given, run training / testing process

        Args:

        sess:
            Tensorflow session object

        graph:
            finalized graph

        save_list:
            list of parameters needs to be saved / restored

        save_dir:
            directory path to save the model.

        evaluate_only:
            if true, only get predict related nodes from graph

        """

        self._sess = sess
        self._graph = graph
        self._save_list = save_list
        self._save_dir = save_dir

        self._saver = tf.train.Saver(save_list)
        self._graph.finalize()
        self._tick_list = []

        with tf.variable_scope("", reuse=True):
            self.global_step = tf.get_variable(
                name="global_step", dtype=tf.int32)

        self.ref_rmse = self._graph.get_tensor_by_name("ref_rmse:0")
        self.inputs = self._graph.get_tensor_by_name("inputs:0")
        self.outputs = self._graph.get_tensor_by_name("outputs:0")

        self.evaluate_only = evaluate_only
        if not evaluate_only:
            self.global_init = self._graph.get_operation_by_name("global_init")
            self.epoch_init = self._graph.get_operation_by_name("epoch_init")
            self.train_op = self._graph.get_operation_by_name("train_op")
            self.global_step_inc = self._graph.get_operation_by_name(
                "global_step_inc")
            self.reference = self._graph.get_tensor_by_name("reference:0")

            self._sess.run(self.global_init)

        self.train_losses = []

    def train(self, epochs=50):
        if self.evaluate_only:
            raise UnableToTrainError("Attempt to run training process on a "
                                     "evaluating model.")
        self._sess.run(self.epoch_init)
        for i in range(epochs):
            try:
                while True:
                    rmse, _ = self._sess.run([self.ref_rmse, self.train_op])
                    self.train_losses.append(rmse)
            except tf.errors.OutOfRangeError:
                self._sess.run(self.epoch_init)
                self._sess.run(self.global_step_inc)

        global_step = self._sess.run(self.global_step)
        print(
            datetime.now(),
            "Training on step ",
            global_step,
            " finished.",
            end="\r")

    def evaluate(self, inputs, reference, msg=""):

        sample = {
            self.inputs: inputs,
            self.reference: reference,
        }

        rmse = self._sess.run(self.ref_rmse, feed_dict=sample)

        return rmse

    def predict(self, inputs):
        sample = {self.inputs: inputs}
        prediction = self._sess.run(self.outputs, feed_dict=sample)

        return prediction

    def add_tick(self, var, func):
        self._tick_list.append([var, func, []])

    def tick(self, update=True, **kwargs):
        g = self._sess.run(self.global_step)
        self._tick_timestamp.append(g)
        for var, func, results in self._tick_list:
            if not update:
                func(self._tick_timestamp, results, **kwargs)
                continue
            if callable(var):
                results.append(var())
            elif isinstance(var, str):
                tensor = self._graph.get_tensor_by_name(var + ":0")
                results.append(self._sess.run(tensor))
            else:
                raise TypeError("unsupported type: ", type(var))
            func(self._tick_timestamp, results, **kwargs)

    def save_model(self):
        print(datetime.now(), ":Saving checkpoints...")
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self._saver.save(
            self._sess, self._save_dir, global_step=self.global_step)

    def load_model(self):
        ckpt = tf.train.latest_checkpoint(self._save_dir)
        if ckpt:
            print(datetime.now(), ": Loading checkpoints from ", ckpt)
            self._saver.restore(self._sess, ckpt)
            return True
        else:
            print(datetime.now(), ": [!] Loading checkpoints fail")
            return False


class UnableToTrainError(Exception):
    pass
