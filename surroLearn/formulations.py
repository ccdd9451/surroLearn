#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

from .utils import tensor_linear_interval_range
from .ops import L2


def linear_regularizer(lr1, lr2, steps):
    def reg(w, *_):
        with tf.name_scope("regularizer"):
            lt = tensor_linear_interval_range(lr1, lr2, steps)
            loss = L2(w, lt)
        lt = tf.identity(lt, name="lambda_scale")

        return loss

    return reg


def classed_rmse(regularizer):

    def Classed_RMSE(prediction, reference, *_):
        """
        Args:

        prediction: Output tensor from the neural network.
        reference:
            Reference tensor from the dataset, with shape
            [batch_size, output_dim].

        Returns:

            RMSE losses with shape [output_dim].

        """

        diff = tf.squared_difference(prediction, reference)
        diff = tf.reduce_mean(diff, axis=1)
        ref_rmse = tf.sqrt(diff)
        ref_rmse = tf.identity(ref_rmse, name="ref_rmse")
        reg = tf.constant(regularizer)
        reg_rmse = tf.div(ref_rmse, reg)
        reg_rmse_sum = tf.reduce_sum(reg_rmse)

        return reg_rmse_sum

    return Classed_RMSE
