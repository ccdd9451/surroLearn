#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def RMSE(prediction, reference, *_):
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
    diff = tf.reduce_mean(diff)
    ref_rmse = tf.sqrt(diff)
    ref_rmse = tf.identity(ref_rmse, name="ref_rmse")

    return ref_rmse


def L2(weights, scale):
    """
    Args:

    weights: List of weights Variables.
    scale: L2 lambda factor tensor.

    Returns:

        L2 regularization term Tensor, scalar.

    """

    weights = map(tf.square, weights)
    weights = map(tf.reduce_sum, weights)
    weights = sum(weights)
    losses = weights * scale

    return losses
