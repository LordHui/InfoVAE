import tensorflow as tf
import numpy as np
import os, time
import subprocess
import argparse
from scipy import misc as misc
from logger import *
from limited_mnist import LimitedMnist
from dataset import *

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-r', '--reg_type', type=str, default='elbo', help='Type of regularization')
parser.add_argument('-g', '--gpu', type=str, default='3', help='GPU to use')
parser.add_argument('-m', '--mi', type=float, default=0.0, help='Information Preference')
parser.add_argument('-s', '--reg_size', type=float, default=50.0, help='Strength of posterior regularization, valid for mmd regularization')
parser.add_argument('-l', '--ll_eval', type=str, default='is', help='is, sampling or both')
args = parser.parse_args()


# python mmd_vae_eval.py --reg_type=elbo --gpu=0 --train_size=1000
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = 200


def make_model_path(name):
    log_path = os.path.join('log', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path

log_path = make_model_path('%s_%.2f_%.2f' % (args.reg_type, args.mi, args.reg_size))


# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    conv = lrelu(conv)
    return conv


def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    conv = tf.nn.relu(conv)
    return conv


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
z_dim = 20
x_dim = [28, 28, 1]
train_x = tf.placeholder(tf.float32, shape=[None]+x_dim)
train_zmean, train_zstddev = encoder(train_x, z_dim)
train_z = train_zmean + tf.multiply(train_zstddev,
                                    tf.random_normal(tf.stack([tf.shape(train_x)[0], z_dim])))
zstddev_logdet = tf.reduce_mean(tf.reduce_sum(2.0 * tf.log(train_zstddev), axis=1))

train_xmean = decoder(train_z)

# Build the computation graph for generating samples
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_xmean = decoder(gen_z, reuse=True)

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
loss_nll_per_sample = -tf.reduce_sum(tf.log(train_xmean) * train_x + tf.log(1 - train_xmean) * (1 - train_x), axis=(1, 2, 3))
loss_nll = tf.reduce_mean(loss_nll_per_sample)


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

trainer = tf.train.AdamOptimizer(4e-5).minimize(loss_all)
logger = RunningAvgLogger(os.path.join(log_path, 'log.txt'), max_step=50)

mnist = MnistDataset(binary=True)

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


def compute_z_logdet(is_train=True):
    z_list = []
    for k in range(50):
        if is_train:
            batch_x = mnist.next_batch(batch_size)
        else:
            batch_x = mnist.next_test_batch(batch_size)
        batch_x = np.reshape(batch_x, [-1]+x_dim)
        z = sess.run(train_z, feed_dict={train_x: batch_x})
        z_list.append(z)
    z_list = np.concatenate(z_list, axis=0)
    cov = np.cov(z_list.T)
    sign, logdet = np.linalg.slogdet(cov)
    return logdet


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tf.global_variables_initializer())

# Start training
# plt.ion()
for i in range(100000):
    batch_x = mnist.next_batch(batch_size)
    batch_x = np.reshape(batch_x, [-1] + x_dim)
    if i < 20000:
        reg_val = 0.01
    else:
        reg_val = 1.0
    _, loss, nll, mmd, elbo, xmean, zlogdet = \
        sess.run([trainer, loss_all, loss_nll, loss_mmd, loss_elbo, train_xmean, zstddev_logdet],
                 feed_dict={train_x: batch_x, reg_coeff: reg_val})
    logger.add_item('loss', loss)
    logger.add_item('nll', nll)
    logger.add_item('mmd', mmd)
    logger.add_item('elbo', elbo)
    logger.add_item('zlogdet', zlogdet)
    if i % 100 == 0:
        print("Iteration %d, nll %.4f, mmd loss %.4f, elbo loss %.4f, zlogdet %f" % (i, nll, mmd, elbo, zlogdet))
        logger.add_item('zlogdet_train', compute_z_logdet(is_train=True))
        logger.add_item('zlogdet_test', compute_z_logdet(is_train=False))
        logger.flush()
    if i % 250 == 0:
        samples_mean = sess.run(gen_xmean, feed_dict={gen_z: np.random.normal(size=(100, z_dim))})
        plots = np.stack([convert_to_display(samples_mean), convert_to_display(np.rint(samples_mean)),
                          convert_to_display(xmean), convert_to_display(np.rint(xmean))], axis=0)
        plots = np.expand_dims(plots, axis=-1)
        plots = convert_to_display(plots)
        misc.imsave(os.path.join(log_path, 'samples%d.png' % i), plots)
