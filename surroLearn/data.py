#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf


class Dataset(object):
    @staticmethod
    def stochastic(inputs, reference):

        inputs = inputs.astype(np.float32)
        reference = reference.astype(np.float32)
        if reference.ndim == 1:
            reference = reference[:, None]

        dat = tf.data.Dataset.from_tensor_slices(tuple(
            inputs, reference)).make_initializable_iterator()
        tf.group(dat.initializer, name="epoch_init")

        inp_t, ref_t = dat.get_next()
        inp_t = tf.identity(inp_t, name="inputs")
        ref_t = tf.identity(ref_t, name="reference")

        return inp_t, ref_t

    @staticmethod
    def shuffle_batch(inputs, reference, batch_size):

        inputs = inputs.astype(np.float32)
        reference = reference.astype(np.float32)
        if reference.ndim == 1:
            reference = reference[:, None]

        dat = tf.data.Dataset.from_tensor_slices(tuple(
            inputs, reference)).shuffle().batch(
                batch_size).make_initializable_iterator()
        tf.group(dat.initializer, name="epoch_init")

        inp_t, ref_t = dat.get_next()
        inp_t = tf.identity(inp_t, name="inputs")
        ref_t = tf.identity(ref_t, name="reference")

        return inp_t, ref_t

    @staticmethod
    def static_tensor(inputs, reference, batch_size):

        inputs = inputs.astype(np.float32)
        reference = reference.astype(np.float32)
        if reference.ndim == 1:
            reference = reference[:, None]

        with tf.variable_scope("extra_tensor"):
            inp_t = tf.convert_to_tensor(inputs, name="inputs")
            ref_t = tf.convert_to_tensor(reference, name="reference")

        return inp_t, ref_t
