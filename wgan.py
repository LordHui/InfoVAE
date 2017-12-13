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

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-g', '--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('-n', '--netname', type=str, default='info_mnist', help='mnist or cifar')
parser.add_argument('-m', '--info_max', type=bool, default=True, help='True or False')
args = parser.parse_args()


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

def conv_large_discriminator(x, data_dims, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)   # 32x32
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)  # 16x16
        conv3 = conv2d_lrelu(conv2, 128, 4, 1)
        conv4 = conv2d_lrelu(conv3, 256, 4, 2)  # 8x8
        conv5 = conv2d_lrelu(conv4, 256, 4, 1)
        conv6 = conv2d_lrelu(conv5, 512, 4, 2)  # 4x4
        conv6 = tf.reshape(conv6, [-1, np.prod(conv6.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv6, 1024)
        fc2 = fc_lrelu(fc1, 1024)
        fc3 = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=tf.identity)
        return fc3


def mlp_discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(x, 1024)
        fc2 = fc_lrelu(fc1, 1024)
        fc3 = fc_lrelu(fc2, 1024)
        fc4 = tf.contrib.layers.fully_connected(fc3, 1, activation_fn=tf.identity)
        return fc4


def conv_large_generator(z, data_dims, range):
    with tf.variable_scope('g_net') as vs:
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, 1024)
        fc3 = fc_bn_relu(fc2, int(data_dims[0]/16)*int(data_dims[1]/16)*512)
        fc3 = tf.reshape(fc3, tf.stack([tf.shape(fc3)[0], int(data_dims[0] / 16), int(data_dims[1] / 16), 512]))
        conv1 = conv2d_t_bn_relu(fc3, 256, 4, 2)
        conv2 = conv2d_t_bn_relu(conv1, 256, 4, 1)
        conv3 = conv2d_t_bn_relu(conv2, 128, 4, 2)
        conv4 = conv2d_t_bn_relu(conv3, 128, 4, 1)
        conv5 = conv2d_t_bn_relu(conv4, 64, 4, 2)
        conv6 = conv2d_t(conv5, data_dims[-1], 4, 2, activation_fn=tf.sigmoid)
        conv6 = conv6 * (range[1] - range[0]) + range[0]
        return conv6


def conv_generator(z, data_dims, range):
    with tf.variable_scope('g_net') as vs:
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, int(data_dims[0]/4)*int(data_dims[1]/4)*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], int(data_dims[0]/4), int(data_dims[1]/4), 128]))
        conv1 = conv2d_t_bn_relu(fc2, 64, 4, 2)
        conv2 = tf.contrib.layers.convolution2d_transpose(conv1, data_dims[-1], 4, 2, activation_fn=tf.sigmoid)
        conv2 = conv2 * (range[1] - range[0]) + range[0]
        return conv2


def mlp_generator(z):
    with tf.variable_scope('g_net') as vs:
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, 1024)
        fc3 = fc_bn_relu(fc2, 1024)
        output = tf.contrib.layers.fully_connected(fc3, 28*28, activation_fn=tf.sigmoid)
        return tf.reshape(output, [-1, 28, 28, 1])


class GenerativeAdversarialNet(object):
    def __init__(self, dataset, name="gan"):
        self.dataset = dataset
        self.data_dims = dataset.data_dims
        print(self.data_dims)
        self.z_dim = 10
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.x = tf.placeholder(tf.float32, [None] + self.data_dims)

        if "large" in name:
            generator = conv_large_generator
            discriminator = conv_large_discriminator
            inference = conv_inference
        else:
            generator = conv_generator
            discriminator = conv_discriminator
            inference = conv_inference

        self.g = generator(self.z, data_dims=self.data_dims, range=dataset.range)
        self.d = discriminator(self.x, data_dims=self.data_dims)
        self.d_ = discriminator(self.g, data_dims=self.data_dims, reuse=True)

        # Variational mutual information maximization
        z_mean, z_stddev = inference(self.g, self.z_dim)
        self.vmi_loss = 0.1 * tf.reduce_sum(tf.log(z_stddev) + tf.square(self.z - z_mean) / tf.square(z_stddev) / 2, axis=1)
        self.vmi_loss = tf.reduce_mean(self.vmi_loss)

        # Gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.g
        d_hat = discriminator(x_hat, data_dims=self.data_dims, reuse=True)

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
        if args.info_max is True:
            self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
                self.g_loss + self.vmi_loss, var_list=self.g_vars+self.i_vars)
        else:
            self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=self.g_vars)
        # self.d_gradient_ = tf.gradients(self.d_loss_g, self.g)[0]
        # self.d_gradient = tf.gradients(self.d_logits, self.x)[0]

        self.merged = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('vmi_loss', self.vmi_loss),
            tf.summary.scalar('d_loss_x', self.d_loss_x),
            tf.summary.scalar('d_loss_g', self.d_loss_g),
            tf.summary.scalar('loss', self.loss)
        ])

        # self.image = tf.summary.image('generated images', self.g, max_images=10)
        self.saver = tf.train.Saver(tf.global_variables())

        self.model_path = "log/%s" % name
        self.fig_path = "%s/fig" % self.model_path

    def visualize(self, batch_size, sess, save_idx):
        bz = np.random.normal(-1, 1, [batch_size, self.z_dim]).astype(np.float32)
        image = sess.run(self.g, feed_dict={self.z: bz})
        num_row = int(math.floor(math.sqrt(batch_size)))
        canvas = np.zeros((self.data_dims[0]*num_row, self.data_dims[1]*num_row, self.data_dims[2]))
        for i in range(num_row):
            for j in range(num_row):
                canvas[i*self.data_dims[0]:(i+1)*self.data_dims[0], j*self.data_dims[1]:(j+1)*self.data_dims[1], :] = \
                    image[i*num_row+j, :, :, :]

        if canvas.shape[-1] == 1:
            misc.imsave("%s/%d.png" % (self.fig_path, save_idx), canvas[:, :, 0])
        else:
            misc.imsave("%s/%d.png" % (self.fig_path, save_idx), canvas)

    def train(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            if os.path.isdir(self.model_path):
                subprocess.call(('rm -rf %s' % self.model_path).split())
            os.makedirs(self.model_path)
            os.makedirs(self.fig_path)
            summary_writer = tf.summary.FileWriter(self.model_path)
            sess.run(tf.global_variables_initializer())
            batch_size = 64

            start_time = time.time()
            for epoch in range(0, 1000):
                batch_idxs = 1093
                for idx in range(0, batch_idxs):
                    if idx % 500 == 0:
                        self.visualize(batch_size, sess, epoch * 2 + idx / 500)

                    bx = self.dataset.next_batch(batch_size)
                    bz = np.random.normal(-1, 1, [batch_size, self.z_dim]).astype(np.float32)
                    d_loss, _, d_loss_g, d_loss_x = sess.run([self.d_loss, self.d_train, self.d_loss_g, self.d_loss_x],
                                                             feed_dict={self.x: bx, self.z: bz})
                    g_loss, _ = sess.run([self.g_loss, self.g_train], feed_dict={self.z: bz})

                    merged = sess.run(self.merged, feed_dict={self.x: bx, self.z: bz})
                    summary_writer.add_summary(merged, epoch * batch_idxs + idx)

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss_x: %.8f, d_loss_g: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss_x, d_loss_g, g_loss))

                save_path = "%s/model" % self.model_path
                if os.path.isdir(save_path):
                    subprocess.call(('rm -rf %s' % save_path).split())
                os.makedirs(save_path)
                self.saver.save(sess, save_path, global_step=epoch)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset = None
    if 'mnist' in args.netname:
        dataset = MnistDataset()
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
