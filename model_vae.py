import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import os, time
import subprocess
import argparse
from scipy import misc as misc
from logger import *
from limited_mnist import LimitedMnist
from dataset import *
from abstract_network import *
from eval_inception import *
from eval_ll import *


# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


# Encoder and decoder use the DC-GAN architecture
# 28 x 28 x 1
def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        conv1 = conv2d_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)   # None x 7 x 7 x 128
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])]) # None x (7x7x128)
        fc1 = fc_lrelu(conv2, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.05)
        mean = tf.maximum(tf.minimum(mean, 10.0), -10.0)
        return mean, stddev


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv1 = conv2d_t_relu(fc2, 64, 4, 2)
        mean = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        mean = tf.maximum(tf.minimum(mean, 0.995), 0.005)
        return mean


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):   # [batch_size, z_dim] [batch_size, z_dim]
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


class VAE:
    def __init__(self, dataset, args):
        self.name = '%s_%d_%.2f_%.2f' % (args.reg_type, args.data_size, args.mi, args.reg_size)
        self.dataset = dataset
        self.data_dims = dataset.data_dims
        self.z_dim = 10
        self.batch_size = 100

        self.train_x = tf.placeholder(tf.float32, shape=[None] + self.data_dims)

        # Build the computation graph for training
        train_zmean, train_zstddev = encoder(self.train_x, self.z_dim)
        self.train_z = train_zmean + tf.multiply(train_zstddev,
                                            tf.random_normal(tf.stack([tf.shape(self.train_x)[0], self.z_dim])))
        train_xmean = decoder(self.train_z)

        # Build the computation graph for generating samples
        self.gen_z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.gen_xmean = decoder(self.gen_z, reuse=True)

        # Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
        true_samples = tf.random_normal(tf.stack([self.batch_size, self.z_dim]))
        self.loss_mmd = compute_mmd(true_samples, self.train_z)

        # ELBO loss divided by input dimensions
        elbo_per_sample = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) +
                                             0.5 * tf.square(train_zmean) - 0.5, axis=1)
        self.loss_elbo = tf.reduce_mean(elbo_per_sample)

        # Negative log likelihood per dimension
        nll_per_sample = -tf.reduce_sum(tf.log(train_xmean) * self.train_x + tf.log(1 - train_xmean) * (1 - self.train_x),
                                             axis=(1, 2, 3))
        self.loss_nll = tf.reduce_mean(nll_per_sample)

        self.reg_coeff = tf.placeholder(tf.float32, shape=[])
        if args.reg_type == 'mmd':
            loss_all = self.loss_nll + (args.reg_size + args.mi - 1.0) * self.loss_mmd + (1.0 - args.mi) * self.loss_elbo
        elif args.reg_type == 'elbo':
            loss_all = self.loss_nll + (1.0 - args.mi) * self.loss_elbo
        elif args.reg_type == 'elbo_anneal':
            loss_all = self.loss_nll + (1.0 - args.mi) * self.loss_elbo * self.reg_coeff
        else:
            loss_all = None
            print("Unknown type")
            exit(-1)

        self.trainer = tf.train.AdamOptimizer(4e-5).minimize(loss_all)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('elbo', self.loss_elbo),
            tf.summary.scalar('nll', self.loss_nll),
            tf.summary.scalar('mmd', self.loss_mmd),
            tf.summary.scalar('loss', loss_all)
        ])

        self.ce_ph = tf.placeholder(tf.float32)
        self.norm1_ph = tf.placeholder(tf.float32)
        self.inception_ph = tf.placeholder(tf.float32)
        self.zlogdet_ph = tf.placeholder(tf.float32)
        self.eval_summary = tf.summary.merge([
            tf.summary.scalar('class_ce', self.ce_ph),
            tf.summary.scalar('norm1', self.norm1_ph),
            tf.summary.scalar('inception_score', self.inception_ph),
            tf.summary.scalar('zlogdet', self.zlogdet_ph),
        ])

        self.train_nll_ph = tf.placeholder(tf.float32)
        self.test_nll_ph = tf.placeholder(tf.float32)
        self.ll_summary = tf.summary.merge([
            tf.summary.scalar('train_nll', self.train_nll_ph),
            tf.summary.scalar('test_nll', self.test_nll_ph)
        ])

        self.log_path, self.fig_path = self.make_model_path()

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.log_path)

        self.inception_evaluator = Classifier()
        self.ll_evaluator = LLEvaluator(self, calibrate=True)

    def get_generator(self, z):
        x_sample = decoder(z, reuse=True)
        return x_sample

    def make_model_path(self):
        log_path = os.path.join('log', self.name)
        if os.path.isdir(log_path):
            subprocess.call(('rm -rf %s' % log_path).split())
        os.makedirs(log_path)
        fig_path = "%s/fig" % log_path
        os.makedirs(fig_path)
        return log_path, fig_path

    def visualize(self, save_idx):
        samples_mean = self.sess.run(self.gen_xmean, feed_dict={self.gen_z: np.random.normal(size=(100, self.z_dim))})
        plots = convert_to_display(samples_mean)
        misc.imsave(os.path.join(self.fig_path, 'samples%d.png' % save_idx), plots)

    def compute_z_logdet(self, is_train=True):
        z_list = []
        for k in range(50):
            if is_train:
                batch_x = self.dataset.next_batch(self.batch_size)
            else:
                batch_x = self.dataset.next_test_batch(self.batch_size)
            batch_x = np.reshape(batch_x, [-1] + x_dim)
            z = self.sess.run(self.train_z, feed_dict={self.train_x: batch_x})
            z_list.append(z)
        z_list = np.concatenate(z_list, axis=0)
        cov = np.cov(z_list.T)
        sign, logdet = np.linalg.slogdet(cov)
        return logdet

    def evaluate_inception(self):
        data_batches = []
        for i in range(20):
            bz = np.random.normal(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            image = self.sess.run(self.gen_xmean, feed_dict={self.gen_z: bz})
            data_batches.append(image)
        class_dist = self.inception_evaluator.class_dist_score(data_batches)
        inception = self.inception_evaluator.inception_score(data_batches)
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
                zlogdet_val = self.compute_z_logdet(is_train=False)
                score_summary = self.sess.run(self.eval_summary,
                                              feed_dict={self.ce_ph: class_dist[0], self.norm1_ph: class_dist[1],
                                                         self.inception_ph: inception, self.zlogdet_ph: zlogdet_val})
                self.summary_writer.add_summary(score_summary, iter)

            bx = self.dataset.next_batch(self.batch_size)
            if iter < 10000:
                reg_val = 0.01
            else:
                reg_val = 1.0
            _, nll, mmd, elbo = \
                self.sess.run([self.trainer, self.loss_nll, self.loss_mmd, self.loss_elbo],
                              feed_dict={self.train_x: bx, self.reg_coeff: reg_val})

            if iter % 1000 == 0:
                print("Iteration %d, time: %4.4f, nll %.4f, mmd %.4f, elbo %.4f" %
                      (iter, time.time() - start_time, nll, mmd, elbo))

            if iter % 100 == 0:
                merged = self.sess.run(self.train_summary, feed_dict={self.train_x: bx})
                self.summary_writer.add_summary(merged, iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

    parser.add_argument('-r', '--reg_type', type=str, default='elbo', help='Type of regularization')
    parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
    parser.add_argument('-m', '--mi', type=float, default=0.0, help='Information Preference')
    parser.add_argument('-s', '--reg_size', type=float, default=50.0,
                        help='Strength of posterior regularization, valid for mmd regularization')
    parser.add_argument('-t', '--data_size', type=int, default=1000)
    args = parser.parse_args()

    # python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset = LimitedMnist(args.data_size, binary=True)
    c = VAE(dataset, args=args)
    c.train()