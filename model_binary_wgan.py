import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import time
import subprocess
import math, os
import scipy.misc as misc
from abstract_network import *
from dataset import *
import argparse
from eval_inception import *
from eval_ll import *


def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv_discriminator(x, data_dims, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return fc2


def conv_inference(x, z_dim, reuse=False):
    with tf.variable_scope('i_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        zmean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        zstddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        zstddev += 0.001
        return zmean, zstddev


def conv_generator(z, data_dims, reuse=False):
    with tf.variable_scope('g_net') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, int(data_dims[0]/4)*int(data_dims[1]/4)*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], int(data_dims[0]/4), int(data_dims[1]/4), 128]))
        conv1 = conv2d_t_bn_relu(fc2, 64, 4, 2)
        conv2 = tf.contrib.layers.convolution2d_transpose(conv1, data_dims[-1], 4, 2, activation_fn=tf.sigmoid)
        conv2 = tf.stop_gradient(tf.round(conv2) - conv2) + conv2
        return conv2


class GenerativeAdversarialNet(object):
    def __init__(self, dataset, name="gan"):
        self.dataset = dataset
        self.data_dims = dataset.data_dims
        self.batch_size = 100
        self.z_dim = 10
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.x = tf.placeholder(tf.float32, [None] + self.data_dims)

        self.generator = conv_generator
        self.discriminator = conv_discriminator
        self.inference = conv_inference

        self.g = self.generator(self.z, data_dims=self.data_dims)
        self.d = self.discriminator(self.x, data_dims=self.data_dims)
        self.d_ = self.discriminator(self.g, data_dims=self.data_dims, reuse=True)

        # Variational mutual information maximization
        z_mean, z_stddev = self.inference(self.g, self.z_dim)
        self.vmi_loss = 0.1 * tf.reduce_sum(tf.log(z_stddev) + tf.square(self.z - z_mean) / tf.square(z_stddev) / 2, axis=1)
        self.vmi_loss = tf.reduce_mean(self.vmi_loss)

        # Gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.g
        d_hat = self.discriminator(x_hat, data_dims=self.data_dims, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
        self.d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

        self.d_loss_x = -tf.reduce_mean(self.d)
        self.d_loss_g = tf.reduce_mean(self.d_)
        self.d_loss = self.d_loss_x + self.d_loss_g + self.d_grad_loss
        self.g_loss = -tf.reduce_mean(self.d_)
        self.loss = self.d_loss + self.g_loss

        self.d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
        self.g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]
        self.i_vars = [var for var in tf.global_variables() if 'i_net' in var.name]
        self.d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
            self.d_loss, var_list=self.d_vars)
        if 'info' in name:
            self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
                self.g_loss + self.vmi_loss, var_list=self.g_vars+self.i_vars)
        else:
            self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=self.g_vars)
        # self.d_gradient_ = tf.gradients(self.d_loss_g, self.g)[0]
        # self.d_gradient = tf.gradients(self.d_logits, self.x)[0]

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('vmi_loss', self.vmi_loss),
            tf.summary.scalar('d_loss_x', self.d_loss_x),
            tf.summary.scalar('d_loss_g', self.d_loss_g),
            tf.summary.scalar('loss', self.loss)
        ])

        self.ce_ph = tf.placeholder(tf.float32)
        self.norm1_ph = tf.placeholder(tf.float32)
        self.inception_ph = tf.placeholder(tf.float32)
        self.eval_summary = tf.summary.merge([
            tf.summary.scalar('class_ce', self.ce_ph),
            tf.summary.scalar('norm1', self.norm1_ph),
            tf.summary.scalar('inception_score', self.inception_ph)
        ])

        self.train_nll_ph = tf.placeholder(tf.float32)
        self.test_nll_ph = tf.placeholder(tf.float32)
        self.ll_summary = tf.summary.merge([
            tf.summary.scalar('train_nll', self.train_nll_ph),
            tf.summary.scalar('test_nll', self.test_nll_ph)
        ])

        # self.image = tf.summary.image('generated images', self.g, max_images=10)
        self.saver = tf.train.Saver(tf.global_variables())

        self.model_path = "log/%s" % name
        self.fig_path = "%s/fig" % self.model_path
        self.make_model_path()

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.model_path)

        self.classifier = Classifier()
        self.ll_evaluator = LLEvaluator(self, calibrate=True)

    def get_generator(self, z):
        x_sample = self.generator(z, data_dims=self.data_dims, reuse=True)
        return x_sample

    def make_model_path(self):
        if os.path.isdir(self.model_path):
            subprocess.call(('rm -rf %s' % self.model_path).split())
        os.makedirs(self.model_path)
        os.makedirs(self.fig_path)

    def visualize(self, save_idx):
        bz = np.random.normal(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        image = self.sess.run(self.g, feed_dict={self.z: bz})
        canvas = convert_to_display(image)
        if canvas.shape[-1] == 1:
            misc.imsave("%s/%d.png" % (self.fig_path, save_idx), canvas[:, :, 0])
        else:
            misc.imsave("%s/%d.png" % (self.fig_path, save_idx), canvas)

    def evaluate_inception(self):
        data_batches = []
        for i in range(20):
            bz = np.random.normal(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            image = self.sess.run(self.g, feed_dict={self.z: bz})
            data_batches.append(image)
        class_dist = self.classifier.class_dist_score(data_batches)
        inception = self.classifier.inception_score(data_batches)
        return class_dist, inception

    def evaluate_ll(self):
        self.ll_evaluator.train()
        train_nll, test_nll = self.ll_evaluator.compute_ll(num_batch=10)
        return train_nll, test_nll

    def train(self):
        start_time = time.time()
        for iter in range(1, 1000000):
            if iter % 1000 == 0:
                train_nll, test_nll = self.evaluate_ll()
                print("Negative log likelihood = %.4f/%.4f" % (train_nll, test_nll))
                ll_summary = self.sess.run(self.ll_summary, feed_dict={self.train_nll_ph: train_nll,
                                                                       self.test_nll_ph: test_nll})
                self.summary_writer.add_summary(ll_summary,  iter)

            if iter % 500 == 0:
                self.visualize(iter)

            if iter % 100 == 0:
                class_dist, inception = self.evaluate_inception()
                score_summary = self.sess.run(self.eval_summary, feed_dict={self.ce_ph: class_dist[0],
                                                                       self.norm1_ph: class_dist[1],
                                                                       self.inception_ph: inception})
                self.summary_writer.add_summary(score_summary, iter)

            bx = self.dataset.next_batch(self.batch_size)
            bz = np.random.normal(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            self.sess.run([self.d_train, self.g_train], feed_dict={self.x: bx, self.z: bz})

            if iter % 1000 == 0:
                d_loss, d_loss_g, d_loss_x, g_loss, i_loss, merged = \
                    self.sess.run([self.d_loss, self.d_loss_g, self.d_loss_x, self.g_loss, self.vmi_loss, self.train_summary],
                                  feed_dict={self.x: bx, self.z: bz})
                print("Iteration %d time: %4.4f, d_loss_x: %.4f, d_loss_g: %.4f, g_loss: %.4f, i_loss: %.4f" \
                      % (iter, time.time() - start_time, d_loss_x, d_loss_g, g_loss, i_loss))

            if iter % 100 == 0:
                merged = self.sess.run(self.train_summary, feed_dict={self.x: bx, self.z: bz})
                self.summary_writer.add_summary(merged, iter)

            if iter % 10000 == 0:
                save_path = "%s/model" % self.model_path
                if os.path.isdir(save_path):
                    subprocess.call(('rm -rf %s' % save_path).split())
                os.makedirs(save_path)
                self.saver.save(self.sess, save_path, global_step=iter//10000)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    # python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

    parser.add_argument('-g', '--gpu', type=str, default='2', help='GPU to use')
    parser.add_argument('-n', '--netname', type=str, default='wgan_mnist', help='mnist or cifar')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset = None
    if 'mnist' in args.netname:
        dataset = MnistDataset(binary=True)
    elif 'cifar' in args.netname:
        dataset = CifarDataset()
    elif 'celeba' in args.netname:
        dataset = CelebADataset(db_path='/ssd_data/CelebA')
    else:
        print("unknown dataset")
        exit(-1)
    c = GenerativeAdversarialNet(dataset, name=args.netname)
    c.train()


# Use histogram between the two distributions as a numerical metric of fitting accuracy
# Even though for many patterns the log likelihood can be accurately estimated, it is a poor indicator of visual appeal
# Use a low dimensional noise vector so that it is impossible for generator to represent the full distribution
# Study the relationship of generator capacity, discriminator capacity vs. quality
# Question: 1. verify that invariant discriminator is the reason for GAN success
# 2. study the relationship between discriminator form and the invariance it encodes
