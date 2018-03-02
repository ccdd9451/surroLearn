#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

def export_graph(path):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(str(path), sess.graph)
        writer.close()
