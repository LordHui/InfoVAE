import math
import numpy as np
import time
import os
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import subprocess
import argparse
from scipy.stats import norm
import seaborn as sns
from scipy import interpolate

plt.ion()
parser = argparse.ArgumentParser()
# python elbo_mog.py --max_reg=0.5 --nll_bound=-3.0 --gpu=0

parser.add_argument('-m', '--max_reg', type=float, default=1.0, help='Maximum coefficient for KL(q(z|x)||p(z))')
parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-i', '--nll_iter', type=int, default=25000, help='Number of iterations for log likelihood evaluation')
parser.add_argument('-r', '--reg', type=str, default='mmd', help='Type of divergence, kl or mmd')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
# loss_nll = tf.maximum(loss_nll, args.nll_bound)

if args.reg == 'kl':
    loss = loss_nll + kl_reg * kl_anneal
else:
    loss = loss_nll + mmd_reg * 500
trainer = tf.train.AdamOptimizer(1e-5).minimize(loss)

train_summary = tf.summary.merge([
    tf.summary.scalar('loss_nll', loss_nll),
    tf.summary.scalar('kl_reg', kl_reg),
    tf.summary.scalar('mme_reg', mmd_reg),
    tf.summary.scalar('loss', loss)
])
true_nll_ph = tf.placeholder(tf.float32)
true_ll_summary = tf.summary.scalar('true_nll', true_nll_ph)


class MoG:
    def __init__(self, centers=None, stddev=0.00000001):
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

def make_model_path(name):
    log_path = os.path.join('log', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path

log_path = make_model_path('%s%.2f' % (args.reg, args.max_reg))
plot_path = os.path.join(log_path, 'plot')
os.makedirs(plot_path)
logger = open(os.path.join(log_path, 'log.txt'), 'w')
writer = tf.summary.FileWriter(log_path, sess.graph)
writer.flush()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.set_tight_layout(True)

num_modes = len(mog.centers)
palette = sns.color_palette("hls", num_modes + 1)
anno_font = 15
label_font = 12
title_font = 12
axis_font = 12
y_lim = 1.6
x_lim = 3.0
sample_size = 40000
num_bins = 100
frame_cnt = 0


def plot_and_fill(ax, x, y, c):
    ax.plot(x, y, color=c)
    ax.fill_between(x, y, color=c, alpha=0.4)


def interp_plot_and_fill(ax, x, y, c):
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(-x_lim, x_lim, 0.001)
    ynew = interpolate.splev(xnew, tck, der=0)
    plot_and_fill(ax, xnew, ynew, c)

next_plot = 200
for i in range(1000000):
    batch_x, _ = mog.sample(batch_size)
    batch_x = batch_x.reshape(-1, x_dim)
    _, total_loss, nll, kl, mmd = sess.run([trainer, loss, loss_nll, kl_reg, mmd_reg],
                                           feed_dict={train_x: batch_x, kl_anneal: 1.0})
    if i % 100 == 0:
        summary = sess.run(train_summary,
                           feed_dict={train_x: batch_x, kl_anneal: 1.0})
        writer.add_summary(summary, i)
        print("Iteration %d: Loss %f, Negative log likelihood is %f, mmd loss is %f, kl loss is %f" % (i, total_loss, nll, mmd, kl))

    if i == next_plot:
        interval = int(next_plot * 0.1 / 100) * 100
        if interval < 200:
            interval = 200
        next_plot += interval
        batch_x, mode_index = mog.sample(sample_size)
        batch_x = batch_x.reshape(-1, x_dim)
        z_samples, x_samples = sess.run([train_zsample, train_xsample], feed_dict={train_x: batch_x})
        samples = sess.run(gen_xsample, feed_dict={gen_z: np.random.normal(size=(sample_size, z_dim))})

        ax[0].cla()
        for modes in range(num_modes):
            indices = np.argwhere(np.equal(mode_index, modes))
            results, edges = np.histogram(z_samples[indices[:, 0], 0], bins=num_bins, range=[-x_lim, x_lim], density=True)
            edges = 0.5 * (edges[1:] + edges[:-1])
            interp_plot_and_fill(ax[0], edges, results, palette[modes])
            # ax[0].hist(z_samples[indices[:, 0], 0], bins=50, color=palette[modes], alpha=0.5, normed=True)
            x_mean = np.mean(z_samples[indices[:, 0], 0])
            y_max = np.max(results)
            horizontal_align = 'center'
            if y_max > 1.3:
                y_max = 1.3
                for i in range(len(results)):
                    if results[i] > 1.3:
                        if modes == 0:
                            x_mean = edges[i] - 0.1
                            horizontal_align = 'right'
                            break
                        else:
                            x_mean = edges[i] + 0.1
                            horizontal_align = 'left'
            if abs(x_mean) < 1:
                x_mean = x_mean / abs(x_mean)
            if abs(x_mean) < 2.5:
                ax[0].text(x_mean, y_max, r'$q_\phi(z|x=%d)$' % mog.centers[modes], fontsize=anno_font,
                           horizontalalignment=horizontal_align, verticalalignment='bottom', color=palette[modes])
        x_axis = np.arange(-x_lim, x_lim, 0.001)
        plot_and_fill(ax[0], x_axis, norm.pdf(x_axis, 0, 1), palette[-1])
        # ax[1].set_yscale('log')
        # ax[1].set_ylim([1e-4, 100])
        ax[0].set_ylim([0, y_lim])
        ax[0].set_xlim([-x_lim, x_lim])
        ax[0].text(0, 0.45, r'$p(z)$', fontsize=anno_font,
                   horizontalalignment='center', verticalalignment='bottom', color=palette[-1])
        ax[0].set_xlabel(r'latent space $z$', fontsize=label_font)
        ax[0].set_ylabel('frequency', fontsize=label_font)
        ax[0].text(0.99, 0.99, 'iteration %08d' % i, transform=ax[0].transAxes,
                   horizontalalignment='right', verticalalignment='top')

        ax[1].cla()
        for modes in range(num_modes):
            # ax[1].axvline(mog.centers[modes], color=sns.color_palette('hls', num_modes+1)[modes])
            indices = np.argwhere(np.equal(mode_index, modes))
            results, edges = np.histogram(x_samples[indices[:, 0], 0], bins=num_bins, range=[-x_lim, x_lim], density=True)
            edges = 0.5 * (edges[1:] + edges[:-1])
            interp_plot_and_fill(ax[1], edges, results, palette[modes])

            x_mean = np.mean(x_samples[indices[:, 0], 0])
            y_max = np.max(results)
            horizontal_align = 'center'
            if y_max > 1.3:
                y_max = 1.3
                for i in range(len(results)):
                    if results[i] > 1.3:
                        if modes == 0:
                            x_mean = edges[i] - 0.1
                            horizontal_align = 'right'
                            break
                        else:
                            x_mean = edges[i] + 0.1
                            horizontal_align = 'left'
            if abs(x_mean) < 1:
                x_mean = x_mean / abs(x_mean)
            if abs(x_mean) < 2.5:
                ax[1].text(x_mean, y_max, r'$p_\theta(x|z)$' + '\n' + r'$z \sim q_\phi(z|x=%d)$' % mog.centers[modes],
                           fontsize=anno_font, horizontalalignment=horizontal_align, verticalalignment='bottom', color=palette[modes])
        results, edges = np.histogram(samples, bins=num_bins, range=[-x_lim, x_lim], density=True)
        y_max = np.max(results[int(3*num_bins/8):int(5*num_bins/8)])
        edges = 0.5 * (edges[1:] + edges[:-1])
        interp_plot_and_fill(ax[1], edges, results, palette[-1])

        # ax[1].set_yscale('log')
        ax[1].set_ylim([0, y_lim])
        ax[1].set_xlim([-x_lim, x_lim])
        ax[1].text(0, y_max, r'$p_\theta(x|z)$' + '\n' + r'$z \sim p(z)$', fontsize=anno_font,
                   horizontalalignment='center', verticalalignment='bottom', color=palette[-1])
        ax[1].set_xlabel(r'input space $x$', fontsize=label_font)
        ax[1].set_ylabel('frequency', fontsize=label_font)
        plt.savefig(os.path.join(plot_path, 'img%04d.png' % frame_cnt))
        frame_cnt += 1
        plt.show()
        plt.pause(0.001)

# ffmpeg -framerate 5 -i plot/img%04d.png -s 640x640  -b 1M -start_number 1 -crf 0 -t 12 output.avi
#