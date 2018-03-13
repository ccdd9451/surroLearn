#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

from tensorflow.contrib.layers import fully_connected


def fully_connected_layer(inputs, num_outputs, scope, trainable_collect,
                          activation_fn):
    outputs = fully_connected(
        inputs=inputs,
        num_outputs=num_outputs,
        activation_fn=activation_fn,
        scope=scope,
    )
    with tf.variable_scope(scope, reuse=True):
        w = tf.get_variable('weights')
        b = tf.get_variable('biases')
        trainable_collect(w, b)

    return outputs


def _stack_fc(inputs, num_hidden, num_outputs, num_level, trainable_collect,
              activation_fn):
    """

    Args:
        inputs: Input tensor from dataset.
        num_hidden: Number of neurons in each layer.
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
        inputs = fully_connected_layer(
            inputs=inputs,
            num_outputs=num_hidden,
            scope=layer_scope,
            trainable_collect=trainable_collect,
            activation_fn=activation_fn,
        )

    output = fully_connected(
        inputs=inputs,
        num_outputs=num_outputs,
        activation_fn=tf.identity,
        scope="layer" + str(num_level))

    output = tf.identity(output, name="outputs")

    with tf.variable_scope("layer" + str(num_level), reuse=True):
        w = tf.get_variable('weights')
        b = tf.get_variable('biases')
        trainable_collect(w, b)

    return output


def stack_fc(activation_fn, num_hidden, num_level, num_outputs=None):
    """
    GraphGen generator
    """
    afns = {
        "lrelu": tf.nn.leaky_relu,
        "relu": tf.nn.relu,
    }
    activation_fn = afns[activation_fn.lower()]

    def graphGen(inputs, references, trainable_collect):
        nonlocal num_outputs
        if not num_outputs:
            num_outputs = int(references.shape[1])
        graph = tf.get_default_graph()
        with graph.as_default():
            _stack_fc(
                inputs=inputs,
                num_hidden=num_hidden,
                num_outputs=num_outputs,
                num_level=num_level,
                trainable_collect=trainable_collect,
                activation_fn=afns,
            )
        return graph

    return graphGen
