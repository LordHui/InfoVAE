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
from dataset import *

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-r', '--reg_type', type=str, default='elbo', help='Type of regularization')
parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-s', '--sparse_ratio', type=float, default=0.1)
args = parser.parse_args()

# python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 200
dataset = MnistDataset()


def make_model_path(name):
    log_path = os.path.join('log/sparsity', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path

log_path = make_model_path('vae%s_%.3f' % (args.reg_type, args.sparse_ratio))


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
        stddev = tf.maximum(stddev, 0.01)
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
        # stddev = tf.contrib.layers.convolution2d_transpose(conv1, 1, 4, 2, activation_fn=tf.sigmoid)
        # stddev = tf.maximum(stddev, 0.01)
        return mean


# Build the computation graph for training
z_dim = 20
x_dim = dataset.data_dims
train_x = tf.placeholder(tf.float32, shape=[None]+x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
train_xr = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_x = decoder(gen_z, reuse=True)


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


# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal(tf.stack([batch_size, z_dim]))
loss_mmd = compute_mmd(true_samples, train_z)

# ELBO loss divided by input dimensions
loss_elbo_per_sample = tf.reduce_sum(-tf.log(train_zstddev) + 0.5 * tf.square(train_zstddev) +
                                     0.5 * tf.square(train_zmean) - 0.5, axis=1)
loss_elbo = tf.reduce_mean(loss_elbo_per_sample)

# Negative log likelihood per dimension
loss_nll_per_sample = 10 * tf.reduce_sum(tf.square(train_x - train_xr), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(loss_nll_per_sample)

loss_sparsity = tf.reduce_mean(tf.reduce_sum(tf.sqrt(1.00001 - train_zstddev) + tf.sqrt(tf.abs(train_zmean) + 0.00001), axis=1))

train_summary = tf.summary.merge([
    tf.summary.scalar('elbo', loss_elbo),
    tf.summary.scalar('mmd', loss_mmd),
    tf.summary.scalar('nll', loss_nll),
    tf.summary.scalar('sparsity', loss_sparsity)
])
sample_summary = tf.summary.merge([
    create_display(tf.slice(train_xr, [0, 0, 0, 0], [100, -1, -1, -1]), name='train_samples'),
    create_display(tf.reshape(gen_x, [100] + x_dim), name='samples')
])

reg_coeff = tf.placeholder(tf.float32, shape=[])
if args.reg_type == 'mmd':
    loss_all = loss_nll + 100 * loss_mmd
elif args.reg_type == 'elbo':
    loss_all = loss_nll + loss_elbo
elif args.reg_type == 'elbo_anneal':
    loss_all = loss_nll + loss_elbo * reg_coeff
else:
    print("Unknown type")
    exit(-1)
loss_all += loss_sparsity * args.sparse_ratio
trainer = tf.train.AdamOptimizer(1e-4).minimize(loss_all)


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples, max_samples=100):
    if max_samples > samples.shape[0]:
        max_samples = samples.shape[0]
    cnt, height, width = int(math.floor(math.sqrt(max_samples))), samples.shape[1], samples.shape[2]
    samples = samples[:cnt*cnt]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(log_path)

# Start training

for i in range(100000):
    batch_x = dataset.next_batch(batch_size)
    if i < 20000:
        reg_val = 0.01
    else:
        reg_val = 1.0
    _, loss, nll, mmd, elbo = sess.run([trainer, loss_all, loss_nll, loss_mmd, loss_elbo],
                                       feed_dict={train_x: batch_x, reg_coeff: reg_val})
    if i % 100 == 0:
        summary_val = sess.run(train_summary, feed_dict={train_x: batch_x, reg_coeff: reg_val})
        writer.add_summary(summary_val, i)
        print("Iteration %d, nll %.4f, mmd loss %.4f, elbo loss %.4f" % (i, nll, mmd, elbo))
    if i % 250 == 0:
        summary_val = sess.run(sample_summary, feed_dict={gen_z: np.random.normal(size=(100, z_dim)),
                                                          train_x: batch_x, reg_coeff: reg_val})
        writer.add_summary(summary_val, i)



