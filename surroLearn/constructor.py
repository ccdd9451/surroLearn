#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from . import data
from .ops import L2


class Constructor(object):
    def __init__(self,
                 graphGen,
                 inputs,
                 references,
                 batch_size,
                 is_training,
                 dataset_split=None,
                 learning_rate=None,
                 random_seed=None):
        """

        Used to construct a graph using on execution phase.


        Args:

        graphGen:
            A function takes (inputs, references, trainable_collect) as
            parameters, returns the core graph represents the complete
            neural network function.

        inputs:
            A Tensor or NdArray with shape (batch_size, input_dim) contains
            all available data for training or A placeholder.

        references:
            Coresponding supervised learning references




        """
        self.graphGen = graphGen
        self.batch_size = batch_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.formulations = {}
        self.devider = data.Devider(inputs, references)

    def regularize_formulate(self, scale=None, function=None):
        """
        Args:

        scale:
            A Tensor or float value, lambda value used when directly performing
            L2 regularization to the graph.

        function:
            A function takes all weights as input, and return the loss as output.

        """

        if scale:

            def reg(w):
                return sum(L2(w, scale))

            self.formulations["regularize"] = reg
        elif function:
            self.formulations["regularize"] = function

        return self.formulations.get("regularize")

    def rmse_loss_formulate(self, function=None):
        """
        Args:

        function:
            A function takes all weights as input, and return the loss as output.

        """

        if function:
            self.formulations["rmse_loss"] = function
        elif not self.formulations.get("rmse_loss"):
            self.formulations["rmse_loss"] = tf.reduce_sum
        return self.formulations["rmse_loss"]

    def main_data_pipe(self):
        return data.Dataset.shuffle_batch(*self.devider.train[:0.75])

    def cross_vaild_pipe(self):
        return data.Dataset.static_tensor(*self.devider.train[0.75:])

    def test_pipe(self):
        return data.Dataset.static_tensor(*self.devider.test.all())

    def training_bake(self):

        weights = []
        biases = []
        others = []

        def trainable_collect(w, b, o=None):
            weights.append(w)
            biases.append(b)
            if o:
                others.append(o)

        # Send all data into tensors
        ti, tr = self.main_data_pipe()
        _, _ = self.cross_vaild_pipe()
        _, _ = self.test_pipe()

        self.graph = self.graphGen(ti, tr, trainable_collect)

        with self.graph.as_default():
            ref_rmse = tf.get_variable("ref_rmse")
            ref_loss = tf.rmse_loss_formulate()(ref_rmse)
            weights_loss = tf.regularize_formulate()(weights)
            tot_loss = ref_loss + weights_loss

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer.minimize(tot_loss, name="train_op")

            epoch = tf.Variable(0, trainable=False, name="global_step")
            tf.assign_add(epoch, 1, name="global_step_inc")

            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            init_d = tf.get_operation_by_name("epoch_init")

            tf.group(init_g, init_l, init_d, name="global_init")

        self.graph.finalize()
        self.save_list = weights + biases + others

        return self.graph
