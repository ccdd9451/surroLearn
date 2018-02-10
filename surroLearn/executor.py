#!/usr/bin/env python
# encoding: utf-8

import os
import tensorflow as tf

from datetime import datetime


class Executor(object):
    def __init__(self, sess, graph, save_list, save_dir):

        self.sess = sess
        self.graph = graph
        self.save_list = save_list
        self.save_dir = save_dir

        self.saver = tf.train.Saver(save_list)
        self.global_step = tf.get_variable("global_step")
        self.ref_loss = tf.get_variable("ref_loss")
        self.inputs = tf.get_variable("inputs")
        self.reference = tf.get_variable("reference")

        self.epoch_init = tf.get_operation_by_name("epoch_init")
        self.train_op = tf.get_operation_by_name("train_op")
        self.global_step_inc = tf.get_operation_by_name("global_step_inc")

    def global_step(self):
        return self.sess.run(self.global_step)

    def train(self, epochs=50):
        self.sess.run(self.epoch_init)
        for i in range(epochs):
            try:
                while True:
                    self.sess.run(self.train_op)
            except tf.errors.OutofRangeError:
                pass

        global_step = self.global_step()
        print(datetime.now(), "Training on step ", global_step, " finished.")

    def evaluate(self, inputs, reference, msg):
        sample = {
            self.inputs: inputs,
            self.reference: reference,
        }

        losses = self.sess.run(self.ref_loss, feed_dict=sample)
        print(datetime.now(), "Evation on ", msg, ": ", losses)

        return losses

    def predict(self, inputs):
        sample = {self.inputs: inputs}
        prediction = self.sess.run(self.reference, feed_dict=sample)

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
