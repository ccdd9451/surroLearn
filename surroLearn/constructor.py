#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from . import data
from .ops import L2


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
                 shuffle_batch_size=256,
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

    def regularize_formulate(self, scale=None, function=None):
        """
        Args:

        scale:
            A Tensor or float value, lambda value used when directly performing
            L2 regularization to the graph.

        function:
            A function f(weights, trainable_collect) takes all weights as input,
            and return the loss as output.

        """

        if scale:

            def reg(w, _=None):
                return sum(L2(w, scale))

            self.formulations["regularize"] = reg
        elif function:
            self.formulations["regularize"] = function

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

            def rmse(outputs, refs, _=None):
                return tf.metrics.root_mean_squared_error(outputs, refs)

            self.formulations["rmse_loss"] = rmse
        return self.formulations["rmse_loss"]

    def main_data_pipe(self):
        return data.Dataset.shuffle_batch(*self.devider.train[:0.75],
                                          self.shuffle_batch_size)

    def cross_vaild_pipe(self):
        return data.Dataset.static_tensor(*self.devider.train[0.75:])

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
        ti, tr = self.main_data_pipe()
        _, _ = self.cross_vaild_pipe()
        _, _ = self.test_pipe()

        ti = tf.identity(ti, name="inputs")
        tr = tf.identity(tr, name="references")
        self.graph = self.graphGen(ti, tr, trainable_collect)

        with self.graph.as_default():
            ref = self.graph.get_tensor_by_name("outputs:0")
            ref_rmse = self.rmse_loss_formulate()(tr, ref)
            ref_rmse = tf.identity(ref_rmse, name="ref_rmse")
            weights_loss = self.regularize_formulate()(weights)
            tot_loss = ref_rmse + weights_loss

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
