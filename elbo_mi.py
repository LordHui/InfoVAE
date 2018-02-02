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
from abstract_network import *

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

# parser.add_argument('-r', '--reg_type', type=str, default='elbo', help='Type of regularization')
parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-n', '--train_size', type=int, default=2000, help='Number of samples for training')
parser.add_argument('-m', '--alpha', type=float, default=50.0)
parser.add_argument('-s', '--beta', type=float, default=51.0)
args = parser.parse_args()


def make_model_path(name):
    log_path = os.path.join('log/elbo_cmi2', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path
log_path = make_model_path('%d_%.2f_%.2f_%.2f' % (args.train_size, args.alpha, args.beta, args.beta-args.alpha))

# python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 200


def encoder(x, z_dim):
    with tf.variable_scope('encoder'):
        conv = conv2d_bn_lrelu(x, 64, 4, 2)   # None x 14 x 14 x 64
        conv = conv2d_bn_lrelu(conv, 64, 4, 1)
        conv = conv2d_bn_lrelu(conv, 128, 4, 2)   # None x 7 x 7 x 128
        conv = conv2d_bn_lrelu(conv, 128, 4, 1)
        conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])]) # None x (7x7x128)
        fc1 = fc_lrelu(conv, 1024)
        mean = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.identity)
        stddev = tf.contrib.layers.fully_connected(fc1, z_dim, activation_fn=tf.sigmoid)
        stddev = tf.maximum(stddev, 0.01)
        return mean, stddev


def decoder(z, reuse=False):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        fc1 = fc_relu(z, 1024)
        fc2 = fc_relu(fc1, 7*7*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], 7, 7, 128]))
        conv = conv2d_t_bn_relu(fc2, 64, 4, 2)
        conv = conv2d_t_bn_relu(conv, 64, 4, 1)
        conv = conv2d_t_relu(conv, 32, 4, 2)
        mean = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 1, activation_fn=tf.sigmoid)
        return mean


# Build the computation graph for training
z_dim = 10
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None] + x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
zstddev_logdet = tf.reduce_mean(tf.reduce_sum(2.0 * tf.log(train_zstddev), axis=1))

train_xr = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_x = decoder(gen_z, reuse=True)

# ELBO loss divided by input dimensions
loss_elbo_per_sample = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) +
                                     0.5 * tf.square(train_zmean) - 0.5, axis=1)
loss_elbo = tf.reduce_mean(loss_elbo_per_sample)

cond_entropy_per_sample = tf.reduce_sum(tf.log(train_zstddev), axis=1)
cond_entropy = tf.reduce_mean(cond_entropy_per_sample)

# Negative log likelihood per dimension
variance = tf.get_variable('coeff', dtype=tf.float32, shape=[], initializer=tf.constant_initializer(0.1))
variance = tf.minimum(tf.maximum(variance, 0.0001), 1.0)
loss_nll_per_sample = tf.reduce_sum(0.5 * tf.log(variance) + tf.square(train_x - train_xr) / 2.0 / variance, axis=(1, 2, 3))
loss_nll = tf.reduce_mean(loss_nll_per_sample)

reg_coeff = tf.placeholder(tf.float32, shape=[])
loss_all = loss_nll + (args.beta * loss_elbo + args.alpha * cond_entropy) * reg_coeff

trainer = tf.train.AdamOptimizer(1e-4).minimize(loss_all)

limited_mnist = LimitedMnist(args.train_size, binary=False)

is_estimator = loss_nll_per_sample + loss_elbo_per_sample


# Evaluate the log likelihood on test data
def compute_log_sum(val):
    min_val = np.min(val, axis=0, keepdims=True)
    return np.mean(min_val - np.log(np.mean(np.exp(-val + min_val), axis=0)))


def compute_nll_by_is(batch_x, sess, verbose=False):
    start_time = time.time()
    nll_list = []
    num_iter = 2000
    for k in range(num_iter):
        nll = sess.run(is_estimator, feed_dict={train_x: batch_x})
        nll_list.append(nll)
        if verbose and (k+1) % 500 == 0:
            print("Iter %d, current value %.4f, time used %.2f" % (
                k, compute_log_sum(np.stack(nll_list)), time.time() - start_time))
    return compute_log_sum(np.stack(nll_list))


def compute_z_logdet(is_train=True):
    z_list = []
    for k in range(40):
        if is_train:
            batch_x = limited_mnist.next_batch(200)
        else:
            batch_x = limited_mnist.test_batch(200)
        batch_x = np.reshape(batch_x, [-1]+x_dim)
        z = sess.run(train_z, feed_dict={train_x: batch_x})
        z_list.append(z)
    z_list = np.concatenate(z_list, axis=0)
    cov = np.cov(z_list.T)
    sign, logdet = np.linalg.slogdet(cov)
    return logdet


train_summary = tf.summary.merge([
    tf.summary.scalar('elbo', loss_elbo),
    tf.summary.scalar('variance', variance),
    tf.summary.scalar('reconstruction', loss_nll),
    tf.summary.scalar('train_nll_elbo', loss_elbo + loss_nll),
    tf.summary.scalar('cond_entropy', cond_entropy),
    tf.summary.scalar('loss', loss_all)
])

sample_summary = tf.summary.merge([
    create_display(tf.slice(train_xr, [0, 0, 0, 0], [100, -1, -1, -1]), 'reconstruction'),
    create_display(tf.reshape(gen_x, [100, 28, 28, 1]), 'samples')
])

nll_ph = tf.placeholder(tf.float32)
logdet_ph = tf.placeholder(tf.float32)
nll_summary = tf.summary.merge([
    tf.summary.scalar('test_nll_is', nll_ph),
])
logdet_summary = tf.summary.merge([
    tf.summary.scalar('zlogdet', logdet_ph),
])

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)

# Start training
# plt.ion()
interval = 5000
for i in range(1000000):
    batch_x = limited_mnist.next_batch(batch_size)
    batch_x = np.reshape(batch_x, [-1] + x_dim)
    if i < 200:
        reg_val = 0.01
    else:
        reg_val = 1.0
    _, elbo, nll = sess.run([trainer, loss_elbo, loss_nll], feed_dict={train_x: batch_x, reg_coeff: reg_val})
    if i % 10 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: batch_x, reg_coeff: reg_val})
        summary_writer.add_summary(summary_val, i)
    if i % 250 == 0:
        summary_val = sess.run(sample_summary, feed_dict={gen_z: np.random.normal(size=(100, z_dim)), train_x: batch_x})
        summary_writer.add_summary(summary_val, i)
        logdet = compute_z_logdet(is_train=False)
        summary_val = sess.run(logdet_summary, feed_dict={logdet_ph: logdet})
        summary_writer.add_summary(summary_val, i)
        print("Iteration %d, nll %.4f, elbo loss %.4f" % (i, nll, elbo))

    if i == interval:
        is_nll = 0.0
        for j in range(40):
            test_data = limited_mnist.test_batch(200)
            is_nll += compute_nll_by_is(test_data, sess=sess, verbose=True)
        is_nll /= 40.0
        summary_val = sess.run(nll_summary, feed_dict={nll_ph: is_nll})
        summary_writer.add_summary(summary_val, i)
        interval = interval * 1.4 + 5000
