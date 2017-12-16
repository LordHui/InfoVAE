import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from abstract_network import *
import tensorflow as tf
from scipy.stats import entropy


def variational_posterior(x, z_dim):
    with tf.variable_scope('vi_posterior'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)   # None x 7 x 7 x 128
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])]) # None x (7x7x128)
        fc1 = fc_lrelu(conv2, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        return mean, stddev


class LikelihoodSmoother:
    def __init__(self):
        pass

    