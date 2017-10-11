import math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class MoG:
    def __init__(self, centers=None, stddev=0.4):
        if centers is not None:
            self.centers = np.array(centers)
        else:
            self.centers = np.array([-1.0, 1.0])
        self.stddev = stddev

    def sample(self, batch_size):
        indices = np.random.choice(range(len(self.centers)), size=batch_size)
        return np.random.normal(size=batch_size) * self.stddev + self.centers[indices], indices


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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
reg = 'kl'
x_dim = 1
z_dim = 1

train_x = tf.placeholder(tf.float32, shape=[None, x_dim])
train_zmean, train_zstddev, train_zsample = encoder(train_x, z_dim)
train_xmean, train_xstddev, train_xsample = decoder(train_zsample)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xmean, gen_xstddev, gen_xsample = decoder(gen_z, reuse=True)


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

if reg == 'kl':
    loss = loss_nll + kl_reg * kl_anneal
else:
    loss = loss_nll + mmd_reg * 500
trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)

batch_size = 500
mog = MoG()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
plt.ion()


c_list = np.array(['r', 'g', 'b', 'm'])
# Start training
for i in range(10000000):
    batch_x, _ = mog.sample(batch_size)
    batch_x = batch_x.reshape(-1, x_dim)
    _, nll, kl, mmd = sess.run([trainer, loss_nll, kl_reg, mmd_reg], feed_dict={train_x: batch_x, kl_anneal: 0.5 - 0.5 * math.exp(-i / 2000.0)})
    if i % 100 == 0:
        print("Iteration %d: Negative log likelihood is %f, mmd loss is %f, kl loss is %f" % (i, nll, mmd, kl))
    if i % 500 == 0:
        batch_x, mode_index = mog.sample(1000)
        batch_x = batch_x.reshape(-1, x_dim)
        z_samples = sess.run(train_zsample, feed_dict={train_x: batch_x})
        samples = sess.run(gen_xsample, feed_dict={gen_z: np.random.normal(size=(1000, z_dim))})
        plt.subplot(1, 2, 1)
        if x_dim == 1:
            plt.hist(batch_x, bins=100, color='r', hold=False)
            plt.hist(samples, bins=100, color='b')
            plt.yscale('log')
            plt.xlim([-3, 3])
        else:
            plt.scatter(samples[:, 0], samples[:, 1], hold=False)
        plt.title('%s: model samples p(x)' % reg)
        plt.subplot(1, 2, 2)
        if z_dim == 1:
            for modes in range(np.max(mode_index) + 1):
                indices = np.argwhere(np.equal(mode_index, modes))
                plt.hist(z_samples[indices[:, 0], 0], bins=100, color=c_list[modes], hold=not modes==0)
        else:
            plt.scatter(z_samples[:, 0], z_samples[:, 1], c=c_list[np.array(mode_index)], hold=False)
        plt.title('%s: latent code q(z)' % reg)
        plt.draw()
        plt.pause(0.001)



