#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

from tensorflow.contrib.layers import fully_connected


def max_out(inputs, num_units):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    num_channels = shape[-1]
    if num_channels % num_units:
        raise ValueError('number fo features({}) in not '
                         'a multiple of num_untis({})'.format(
                             num_channels, num_units))
    shape[-1] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs


def layer_max_out(inputs, num_outputs, scope, trainable_collect, num_units=10):
    outputs = fully_connected(
        inputs=inputs,
        num_outputs=num_outputs,
        activation_fn=lambda inputs: max_out(inputs, num_units),
        scope=scope)
    with tf.variable_scope(scope, reuse=True):
        w = tf.get_variable('weights')
        b = tf.get_variable('biases')
        trainable_collect(w, b)

    return outputs


def stack_max_out(inputs, num_cells, num_units, num_outputs, num_level,
                   trainable_collect):
    """

    Args:
        inputs: Input tensor from dataset.
        num_cells: Number of neurons in each layer.
        num_units: Units output after maxout activation function.
        num_outputs: Width of output tensor.
        num_level:
            Neural network levels, defined as number of hidden
            layer + 1.
        trainable_collect:
            An callable function which takes two parameters
            F(weights, biases).

    Returns:
        The output tensor of the graph.

    Core network of a maxout layer stack.

    """
    layers = ["layer" + str(i) for i in range(num_level)]
    for layer_scope in layers:
        inputs = layer_max_out(
            inputs=inputs,
            num_outputs=num_cells,
            scope=layer_scope,
            trainable_collect=trainable_collect,
            num_units=num_units)

    output = fully_connected(
        inputs=inputs,
        num_outputs=num_outputs,
        activation_fn=tf.identity,
        scope="layer" + str(num_level))

    with tf.variable_scope("layer"+str(num_level), reuse=True):
        w = tf.get_variable('weights')
        b = tf.get_variable('biases')
        trainable_collect(w, b)

    return output
