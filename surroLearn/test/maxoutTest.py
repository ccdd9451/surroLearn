#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

import surroLearn as sl


class GraphTest(tf.test.TestCase):
    def test_CreateGraph(self):
        g = tf.Graph()
        with self.test_session(graph=g):
            inputs = tf.constant(value=1, dtype=tf.float32, shape=(123, 10))

            weights_var = []
            biases_var = []

            def collect_trainables(weights, biases):
                weights_var.append(weights)
                biases_var.append(biases)

            outputs = sl.maxout.stack_max_out(
                inputs=inputs,
                num_cells=100,
                num_units=10,
                num_outputs=4,
                num_level=6,
                trainable_collect=collect_trainables)

            self.assertTrue(len(weights_var) == 7, str(len(weights_var)))
            self.assertTrue(len(biases_var) == 7, str(len(biases_var)))
            self.assertEqual(outputs.get_shape().as_list(), [123, 4],
                             str(outputs.get_shape().as_list()))


def main():
    tf.test.main()


if __name__ == "__main__":
    main()
