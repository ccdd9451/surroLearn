#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf


class Dataset(object):
    @staticmethod
    def stochastic(inputs, reference):

        dat = tf.data.Dataset.from_tensor_slices(tuple(
            inputs, reference)).make_initializable_iterator()
        tf.group(dat.initializer, name="epoch_init")

        inp_t, ref_t = dat.get_next()
        inp_t = tf.identity(inp_t, name="inputs")
        ref_t = tf.identity(ref_t, name="reference")

        return inp_t, ref_t

    @staticmethod
    def shuffle_batch(inputs, reference, batch_size):

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

        with tf.variable_scope("extra_tensor"):
            inp_t = tf.convert_to_tensor(inputs, name="inputs")
            ref_t = tf.convert_to_tensor(reference, name="reference")

        return inp_t, ref_t


class Devider(object):
    class Barrel(object):
        def __init__(self, inp, ref):
            self.inp = inp
            self.ref = ref
            self.size = inp.shape[0]

        def __getitem__(self, sl):
            start = int(sl.start * self.size)
            stop = int(sl.stop * self.size)

            return (self.inp[start:stop, :], self.ref[start:stop, :])

        def all(self):
            return (self.inp, self.ref)

        def lim_amount(self, start, stop, size):
            start = int(start * self.size)
            stop = int(stop * self.size)

            return (self.inp[start:stop, :][:size, :],
                    self.ref[start:stop, :][:size, :])

    def __init__(self, inputs, reference, shuffle_range=0.85, seed=None):

        self.inputs = inputs.astype(np.float32)
        self.reference = reference.astype(np.float32)
        if reference.ndim == 1:
            reference = reference[:, None]

        data_size = self.inputs.shape[0]
        self.test_cut = int(data_size * shuffle_range)

        if seed:
            np.random.seed(seed)

        self.shuffled_indicies = np.arange(self.test_cut)
        np.random.shuffle(self.shuffled_indicies)

        self.train = Devider.Barrel(self.inputs[self.shuffled_data, :],
                                    self.reference[self.shuffled_data, :])
        self.test = Devider.Barrel(self.inputs[self.test_cut:, :],
                                   self.reference[self.test_cut:, :])
