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

        self.sess = sess
        self.graph = graph
        self.save_list = save_list
        self.save_dir = save_dir

        self.saver = tf.train.Saver(save_list)
        self.graph.finalize()

        with tf.variable_scope("", reuse=True):
            self.global_step = tf.get_variable(name="global_step", dtype=tf.int32)

        self.ref_rmse = self.graph.get_tensor_by_name("ref_rmse:0")
        self.inputs = self.graph.get_tensor_by_name("inputs:0")
        self.outputs = self.graph.get_tensor_by_name("outputs:0")

        self.evaluate_only = evaluate_only
        if not evaluate_only:
            self.global_init = self.graph.get_operation_by_name("global_init")
            self.epoch_init = self.graph.get_operation_by_name("epoch_init")
            self.train_op = self.graph.get_operation_by_name("train_op")
            self.global_step_inc = self.graph.get_operation_by_name("global_step_inc")
            self.reference = self.graph.get_tensor_by_name("reference:0")

            self.sess.run(self.global_init)


    def train(self, epochs=50):
        if self.evaluate_only:
            raise UnableToTrainError("Attempt to run training process on a "
                                     "evaluating model.")
        self.sess.run(self.epoch_init)
        print("Start Batch Training, losses:")
        for i in range(epochs):
            try:
                while True:
                    rmse, _ = self.sess.run([self.ref_rmse, self.train_op])
                    print(rmse, end="\t")
            except tf.errors.OutOfRangeError:
                self.sess.run(self.epoch_init)
                self.sess.run(self.global_step_inc)

        global_step = self.sess.run(self.global_step)
        print(datetime.now(), "Training on step ", global_step, " finished.")

    def evaluate(self, inputs, reference, msg=""):

        sample = {
            self.inputs: inputs,
            self.reference: reference,
        }

        rmse = self.sess.run(self.ref_rmse, feed_dict=sample)
        print(datetime.now(), "Evaluating on ", msg, ": ", rmse)

        return rmse

    def predict(self, inputs):
        sample = {self.inputs: inputs}
        prediction = self.sess.run(self.outputs, feed_dict=sample)

        return prediction

    def save_model(self):
        print(datetime.now(), ":Saving checkpoints...")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.saver.save(self.sess, self.save_dir, global_step=self.global_step)

    def load_model(self):
        ckpt = tf.train.latest_checkpoint(self.save_dir)
        if ckpt:
            print(datetime.now(), ": Loading checkpoints from ", ckpt)
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            print(datetime.now(), ": [!] Loading checkpoints fail")
            return False


class UnableToTrainError(Exception):
    pass
