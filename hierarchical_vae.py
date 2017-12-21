import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import os, time
import subprocess
import argparse
from scipy import misc as misc
from logger import *
from abstract_network import *
from dataset import *
from limited_mnist import LimitedMnist


parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-r', '--reg_type', type=str, default='elbo', help='Type of regularization')
parser.add_argument('-g', '--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('-m', '--mi', type=float, default=0.0, help='Information Preference')
parser.add_argument('-s', '--reg_size', type=float, default=50.0, help='Strength of posterior regularization, valid for mmd regularization')
parser.add_argument('-z', '--zdim', type=int, default=10, help='Dimensionality of z')
parser.add_argument('-n', '--name', type=str, default='', help='Run name')
args = parser.parse_args()


# python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 100


def make_model_path(name):
    log_path = os.path.join('log', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path

if len(args.name) == 0:
    args.name = 'h%s_%.1f_%.1f' % (args.reg_type, args.mi, args.reg_size)
log_path = make_model_path(args.name)


# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def pre_encoder(x):
    with tf.variable_scope('pre_encoder'):
        conv1 = conv2d_lrelu(x, 64, 4, 1)
        conv2 = conv2d_lrelu(conv1, 8, 4, 1)
        return conv2


def pre_decoder(z, reuse=False):
    with tf.variable_scope('pre_decoder') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_t_relu(z, 64, 4, 1)
        conv2 = tf.contrib.layers.convolution2d(conv1, 1, 4, 1, activation_fn=tf.sigmoid)
        return conv2


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
        return mean


# Build the computation graph for training
z_dim = args.zdim
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None]+x_dim)
train_xz = pre_encoder(train_x)
train_zmean, train_zstddev = encoder(train_xz, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))

train_xzr = decoder(train_z)
train_xr = pre_decoder(train_xzr)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xz = decoder(gen_z, reuse=True)
gen_x = pre_decoder(gen_xz, reuse=True)


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
pre_reconstruction = 10.0 * tf.reduce_mean(tf.reduce_sum(tf.square(train_xzr - train_xz), axis=(1, 2, 3)))
reconstruction = 10.0 * tf.reduce_mean(tf.reduce_sum(tf.square(train_xr - train_x), axis=(1, 2, 3)))
loss_nll = pre_reconstruction + reconstruction

reg_coeff = tf.placeholder(tf.float32, shape=[])
if args.reg_type == 'mmd':
    loss_all = loss_nll + (args.reg_size + args.mi - 1.0) * loss_mmd + (1.0 - args.mi) * loss_elbo
elif args.reg_type == 'elbo':
    loss_all = loss_nll + (1.0 - args.mi) * loss_elbo
elif args.reg_type == 'elbo_anneal':
    loss_all = loss_nll + (1.0 - args.mi) * loss_elbo * reg_coeff
else:
    print("Unknown type")
    exit(-1)

trainer = tf.train.AdamOptimizer(1e-4).minimize(loss_all)
train_summary = tf.summary.merge([
    tf.summary.scalar('mmd', loss_mmd),
    tf.summary.scalar('elbo', loss_elbo),
    tf.summary.scalar('pre-reconstruction', pre_reconstruction),
    tf.summary.scalar('reconstruction', reconstruction),
    tf.summary.scalar('loss', loss_all)
])

xr_slices = [tf.reshape(train_xz[:, :, :, i:i+1], [100, 28, 28, 1]) for i in range(8)]
img_summary = tf.summary.merge([
    create_multi_display([tf.reshape(train_xr, [100, 28, 28, 1])] + xr_slices, 'train'),
    create_display(tf.reshape(gen_x, [100, 28, 28, 1]), 'samples')
])
dataset = MnistDataset()

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path)

# Start training
# plt.ion()
for i in range(100000):
    batch_x = dataset.next_batch(batch_size)
    if i < 20000:
        reg_val = 0.01
    else:
        reg_val = 1.0
    _, loss, nll, mmd, elbo, rec, pre_rec = \
        sess.run([trainer, loss_all, loss_nll, loss_mmd, loss_elbo, reconstruction, pre_reconstruction], feed_dict={train_x: batch_x, reg_coeff: reg_val})
    if i % 100 == 0:
        print("Iteration %d, nll %.4f, mmd loss %.4f, elbo loss %.4f, rec %.4f, pre-rec %.4f" % (i, nll, mmd, elbo, rec, pre_rec))
        summary_writer.add_summary(sess.run(train_summary, feed_dict={train_x: batch_x, reg_coeff: reg_val}), i)
    if i % 250 == 0:
        bz = np.random.normal(size=(100, z_dim))
        samples_mean = sess.run(gen_x, feed_dict={gen_z: bz})
        plots = convert_to_display(samples_mean)
        misc.imsave(os.path.join(log_path, 'samples%d.png' % i), plots)
        summary_writer.add_summary(sess.run(img_summary, feed_dict={train_x: batch_x, reg_coeff: reg_val, gen_z: bz}), i)


