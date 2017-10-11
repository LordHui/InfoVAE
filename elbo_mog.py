import math
import numpy as np
import time
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import subprocess
import argparse

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-m', '--max_reg', type=float, default=1.0, help='Maximum coefficient for KL(q(z|x)||p(z))')
parser.add_argument('-n', '--nll_bound', type=float, default=-3.0, help='Lower bound on nll')
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-i', '--nll_iter', type=int, default=25000, help='Number of iterations for log likelihood evaluation')
args = parser.parse_args()


# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc


def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc


# Encoder and decoder use the DC-GAN architecture
def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        temp = fc_lrelu(x, 200)
        temp = fc_lrelu(temp, 200)
        mean = tf.contrib.layers.fully_connected(temp, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(temp, z_dim, activation_fn=tf.sigmoid)
        stddev += 0.0005
        sample = mean + stddev * tf.random_normal(tf.stack([tf.shape(x)[0], z_dim]))
        return mean, stddev, sample


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        temp = fc_lrelu(z, 200)
        temp = fc_lrelu(temp, 200)
        mean = tf.contrib.layers.fully_connected(temp, 1, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(temp, 1, activation_fn=tf.sigmoid)
        stddev += 0.0005
        sample = mean + stddev * tf.random_normal(tf.stack([tf.shape(z)[0], 1]))
        return mean, stddev, sample


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
reg = 'kl'

x_dim = 1
z_dim = 1

train_x = tf.placeholder(tf.float32, shape=[None, x_dim])
train_zmean, train_zstddev, train_zsample = encoder(train_x, z_dim)
train_xmean, train_xstddev, train_xsample = decoder(train_zsample)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xmean, gen_xstddev, gen_xsample = decoder(gen_z, reuse=True)

sample_nll = tf.square(train_x - gen_xmean) / (2 * tf.square(gen_xstddev)) + tf.log(gen_xstddev)
sample_nll += 0.5 * math.log(2 * np.pi)
sample_nll = tf.reduce_sum(sample_nll, axis=1)     # negative log likelihood per dimension

# KL Regularization
kl_reg = -tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) + 0.5 * tf.square(train_zmean) - 0.5
kl_reg = tf.reduce_mean(tf.reduce_sum(kl_reg, axis=1))
kl_anneal = tf.placeholder(tf.float32)

# MMD Regularization
true_samples = tf.random_normal(tf.stack([500, z_dim]))
mmd_reg = compute_mmd(true_samples, train_zsample)

# Log likelihood loss
loss_nll = 0.5 * math.log(2 * math.pi) + tf.log(train_xstddev) + \
    tf.square(train_xmean - train_x) / (2 * tf.square(train_xstddev))
loss_nll = tf.reduce_mean(tf.reduce_sum(loss_nll, axis=1))
loss_nll = tf.maximum(loss_nll, args.nll_bound)

if reg == 'kl':
    loss = loss_nll + kl_reg * kl_anneal
else:
    loss = loss_nll + mmd_reg * 500
trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)

train_summary = tf.summary.merge([
    tf.summary.scalar('loss_nll', loss_nll),
    tf.summary.scalar('kl_reg', kl_reg),
    tf.summary.scalar('mme_reg', mmd_reg),
    tf.summary.scalar('loss', loss)
])
true_nll_ph = tf.placeholder(tf.float32)
true_ll_summary = tf.summary.scalar('true_nll', true_nll_ph)


class MoG:
    def __init__(self, centers=None, stddev=0.5):
        if centers is not None:
            self.centers = np.array(centers)
        else:
            self.centers = np.array([-1.0, 1.0])
        self.stddev = stddev

    def sample(self, batch_size):
        indices = np.random.choice(range(len(self.centers)), size=batch_size)
        return np.random.normal(size=batch_size) * self.stddev + self.centers[indices], indices


batch_size = 500
mog = MoG()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())


def compute_log_sum(val):
    min_val = np.min(val, axis=0, keepdims=True)
    return np.mean(min_val - np.log(np.mean(np.exp(-val + min_val), axis=0)))


def compute_true_nll():
    start_time = time.time()
    avg_nll = []
    for i in range(5):
        batch_x, _ = mog.sample(batch_size)
        batch_x = batch_x.reshape(-1, x_dim)
        nll_list = []
        num_iter = args.nll_iter
        for k in range(num_iter):
            random_z = np.random.normal(size=[batch_size, z_dim])
            nll = sess.run(sample_nll, feed_dict={train_x: batch_x, gen_z: random_z})
            nll_list.append(nll)
            if k % 50 == 0:
                print("iter %d, current value %.4f, time used %.2f" % (k, compute_log_sum(np.stack(nll_list)), time.time() - start_time))
        nll = compute_log_sum(np.stack(nll_list))
        avg_nll.append(nll)
        print("likelihood importance sampled = %.4f, time used %.2f" % (nll, time.time() - start_time))
    return np.mean(avg_nll)


def make_model_path(name):
    log_path = os.path.join('log', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path

log_path = make_model_path('kl%.2f-%.2f' % (args.max_reg, args.nll_bound))
plot_path = os.path.join(log_path, 'plot')
os.makedirs(plot_path)
logger = open(os.path.join(log_path, 'log.txt'), 'w')
writer = tf.summary.FileWriter(log_path, sess.graph)
writer.flush()


c_list = np.array(['r', 'g', 'b', 'm'])
# Start training

for i in range(1, 10000000):
    batch_x, _ = mog.sample(batch_size)
    batch_x = batch_x.reshape(-1, x_dim)
    if i < 2000:
        kl_ratio = 0.01
    else:
        kl_ratio = args.max_reg
    _, total_loss, nll, kl, mmd = sess.run([trainer, loss, loss_nll, kl_reg, mmd_reg],
                                           feed_dict={train_x: batch_x, kl_anneal: kl_ratio})
    if i % 100 == 0:
        summary = sess.run(train_summary,
                           feed_dict={train_x: batch_x, kl_anneal: kl_ratio})
        writer.add_summary(summary, i)
        print("Iteration %d: Loss %f, Negative log likelihood is %f, mmd loss is %f, kl loss is %f" % (i, total_loss, nll, mmd, kl))
    if i % 1000 == 0:
        batch_x, mode_index = mog.sample(1000)
        batch_x = batch_x.reshape(-1, x_dim)
        z_samples = sess.run(train_zsample, feed_dict={train_x: batch_x})
        samples = sess.run(gen_xsample, feed_dict={gen_z: np.random.normal(size=(1000, z_dim))})
        if x_dim == 1:
            plt.hist(batch_x, bins=100, color='r', hold=False)
            plt.hist(samples, bins=100, color='b')
            plt.yscale('log')
            plt.xlim([-3, 3])
        else:
            plt.scatter(samples[:, 0], samples[:, 1], hold=False)
        plt.title('%s: model samples p(x)' % reg)
        plt.savefig(os.path.join(plot_path, 'model_sample%d.png' % i))

        if z_dim == 1:
            for modes in range(np.max(mode_index) + 1):
                indices = np.argwhere(np.equal(mode_index, modes))
                plt.hist(z_samples[indices[:, 0], 0], bins=100, color=c_list[modes], hold=not modes == 0)
        else:
            plt.scatter(z_samples[:, 0], z_samples[:, 1], c=c_list[np.array(mode_index)], hold=False)
        plt.title('%s: latent code q(z)' % reg)
        plt.savefig(os.path.join(plot_path, 'latent_code%d.png', i))

        true_nll = compute_true_nll()
        logger.write('%d %f %f %f %f\n' % (i, nll, kl, mmd, true_nll))
        logger.flush()
        summary = sess.run(true_ll_summary, feed_dict={true_nll_ph: true_nll})
        writer.add_summary(summary, i)
        print("True log likelihood is %f" % true_nll)
