#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from . import data
from functools import lru_cache


class Constructor(object):
    """

    Tensorflow graph constructor. The default behavior as following:

    c = Constructor(graphGen, inputs, references, random_seed)

    ``optional
    c.regularize_formulate(function=f)
    c.rmse_loss_formulate(function=f)
    ``

    graph = c.training_bake()

    """

    def __init__(self,
                 graphGen,
                 inputs,
                 references,
                 shuffle_batch_size=1024,
                 learning_rate=0.001,
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
        self.formulations = {}
        self.learning_rate = learning_rate
        self.shuffle_batch_size = shuffle_batch_size
        self.devider = data.Devider(inputs, references, seed=random_seed)

    def regularize_formulate(self, scale=None):
        """
        Args:

        scale:
            A Tensor or float value, lambda value used when directly performing
            L2 regularization to the graph.

        function:
            A function f(weights, trainable_collect) takes all weights as input,
            and return the loss as output.

        """

        if callable(scale):
            self.formulations["regularize"] = scale
        elif scale:
            from .ops import L2

            def reg(w, _=None):
                return L2(w, scale)

            self.formulations["regularize"] = reg

        f = self.formulations.get("regularize", None)
        if f:
            return f
        else:
            return lambda *x: 0

    def rmse_loss_formulate(self, function=None):
        """
        Args:

        function:
            A function f(predicts, references, trainable_collect) takes all weights as input,
            and return the loss as output.

        """

        if function:
            self.formulations["rmse_loss"] = function
        elif not self.formulations.get("rmse_loss"):
            from .ops import RMSE
            self.formulations["rmse_loss"] = RMSE
        return self.formulations["rmse_loss"]

    @lru_cache()
    def main_data_pipe(self):
        return data.Dataset.shuffle_batch(*self.devider.train[:0.75],
                                          self.shuffle_batch_size)

    @lru_cache()
    def cross_vaild_pipe(self):
        return data.Dataset.static_tensor(*self.devider.train[0.75:])

    @lru_cache()
    def test_pipe(self):
        return data.Dataset.static_tensor(*self.devider.test.all())

    def training_bake(self):

        weights = []
        biases = []
        others = []

        def trainable_collect(w=None, b=None, o=None):
            if w:
                weights.append(w)
            if b:
                biases.append(b)
            if o:
                others.append(o)

        # Send all data into tensors
        _, _ = self.main_data_pipe()
        _, _ = self.cross_vaild_pipe()
        _, _ = self.test_pipe()
        pipe = tf.constant("train", name="pipe")
        compare = {
            tf.equal(pipe, "train"): self.main_data_pipe,
            tf.equal(pipe, "cross_valid"): self.cross_vaild_pipe,
            tf.equal(pipe, "test"): self.test_pipe,
        }
        ti, tr = tf.case(compare, exclusive=True)
        ti = tf.identity(ti, name="inputs")
        tr = tf.identity(tr, name="references")
        self.graph = self.graphGen(ti, tr, trainable_collect)

        with self.graph.as_default():
            epoch = tf.get_variable(name="global_step", initializer=0)
            tf.assign_add(epoch, 1, name="global_step_inc")

            ref = self.graph.get_tensor_by_name("outputs:0")
            ref_rmse = self.rmse_loss_formulate()(tr, ref)
            weights_loss = self.regularize_formulate()(weights)
            tot_loss = ref_rmse + weights_loss

            ref_rmse = tf.identity(ref_rmse, name="ref_rmse")

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer.minimize(tot_loss, name="train_op")

            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            init_d = self.graph.get_operation_by_name("epoch_init")

            tf.group(init_g, init_l, init_d, name="global_init")

        self.save_list = weights + biases + others

        return self.graph
