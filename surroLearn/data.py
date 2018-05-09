#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
"""

    All dataset must be composed as two matrices (X, Y) whose shape
    is [batch_size, features]

"""


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

        dat = tf.data.Dataset.from_tensor_slices((inputs, reference)).shuffle(
            inputs.shape[0]).batch(batch_size).make_initializable_iterator()
        tf.group(dat.initializer, name="epoch_init")

        inp_t, ref_t = dat.get_next()
        inp_t = tf.identity(inp_t, name="inputs")
        ref_t = tf.identity(ref_t, name="reference")

        return inp_t, ref_t

    @staticmethod
    def static_tensor(inputs, reference):

        with tf.variable_scope("extra_tensor"):
            inp_t = tf.convert_to_tensor(inputs, name="inputs")
            ref_t = tf.convert_to_tensor(reference, name="reference")

        return inp_t, ref_t

    @staticmethod
    def restricted_opt_container(inputs, reference, length):
        shape = inputs.shape[1]
        mins = inputs.min(axis=0)
        maxs = inputs.max(axis=0)
        ru = tf.random_uniform((length, shape))

        with tf.variable_scope("", reuse=False):
            inp_t = tf.get_variable(
                name="opt_container",
                initializer=ru * (maxs - mins) + mins,
            )
            ref_t = tf.zeros([length, 1])

        minvals = mins[None, :].repeat(length, 0)
        maxvals = maxs[None, :].repeat(length, 0)

        inp_t = tf.clip_by_value(inp_t, minvals, maxvals)
        inp_t = tf.identity(inp_t, name="inputs")
        ref_t = tf.identity(ref_t, name="reference")
        return inp_t, ref_t


class Devider(object):
    """

        Devide dataset into train/test dataset first. After that, one
        can choose to get all data from the dataset, or just picking
        up part of the data

    """

    class Barrel(object):
        def __init__(self, inp, ref):
            self.inp = inp
            self.ref = ref
            self.size = inp.shape[0]

        def __getitem__(self, sl):
            start = int(sl.start * self.size) if sl.start else None
            stop = int(sl.stop * self.size) if sl.stop else None

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
        if self.reference.ndim == 1:
            self.reference = self.reference[:, None]

        data_size = self.inputs.shape[0]
        self.test_cut = int(data_size * shuffle_range)

        if seed:
            np.random.seed(seed)

        self.shuffled_indicies = np.arange(self.test_cut)
        np.random.shuffle(self.shuffled_indicies)

        self.train = Devider.Barrel(self.inputs[self.shuffled_indicies, :],
                                    self.reference[self.shuffled_indicies, :])
        self.test = Devider.Barrel(self.inputs[self.test_cut:, :],
                                   self.reference[self.test_cut:, :])


def load(filename):
    from pickle import load

    with open(filename, "rb") as f:
        return load(f)


def compatible_load(filename):
    d = load(filename)
    return d["X"], d["Y"]


def unittest_sample():
    inputs = np.random.randn(100, 10)
    references = np.random.randn(100)

    return inputs, references
