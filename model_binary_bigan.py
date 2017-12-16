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

parser = argparse.ArgumentParser()
# python coco_transfer2.py --db_path=../data/coco/coco_seg_transfer40_30_299 --batch_size=64 --gpu='0' --type=mask

parser.add_argument('-g', '--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('-n', '--netname', type=str, default='bigan_mnist', help='mnist or cifar')
args = parser.parse_args()


def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


def conv_discriminator(x, z, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1_x = fc_lrelu(conv2, 512)
        fc1_z = fc_lrelu(z, 512)
        fc1 = tf.concat([fc1_x, fc1_z], axis=1)
        fc2 = fc_lrelu(fc1, 1024)
        fc3 = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=tf.identity)
        return fc3


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


def conv_generator(z, data_dims):
    with tf.variable_scope('g_net') as vs:
        fc1 = fc_bn_relu(z, 1024)
        fc2 = fc_bn_relu(fc1, int(data_dims[0]/4)*int(data_dims[1]/4)*128)
        fc2 = tf.reshape(fc2, tf.stack([tf.shape(fc2)[0], int(data_dims[0]/4), int(data_dims[1]/4), 128]))
        conv1 = conv2d_t_bn_relu(fc2, 64, 4, 2)
        xmean = tf.contrib.layers.convolution2d_transpose(conv1, data_dims[-1], 4, 2, activation_fn=tf.sigmoid)
        xsample = tf.stop_gradient(tf.round(xmean) - xmean) + xmean
        return xmean, xsample


class GenerativeAdversarialNet(object):
    def __init__(self, dataset, name="gan"):
        self.dataset = dataset
        self.data_dims = dataset.data_dims
        print(self.data_dims)
        self.z_dim = 10
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.x = tf.placeholder(tf.float32, [None] + self.data_dims)

        generator = conv_generator
        discriminator = conv_discriminator
        inference = conv_inference

        self.x_mean, self.x_sample = generator(self.z, data_dims=self.data_dims)
        self.z_mean, self.z_stddev = inference(self.x, z_dim=self.z_dim)
        self.z_sample = self.z_mean + tf.multiply(self.z_stddev,
                                             tf.random_normal(tf.stack([tf.shape(self.x)[0], self.z_dim])))
        self.dx = discriminator(self.x, self.z_sample)
        self.dz = discriminator(self.x_sample, self.z, reuse=True)

        # Variational mutual information maximization
        self.z_recon = 0.1 * tf.reduce_sum(tf.log(self.z_stddev) + tf.square(self.z - self.z_mean) / tf.square(self.z_stddev) / 2, axis=1)
        self.z_recon = tf.reduce_mean(self.z_recon)

        self.x_recon = -tf.reduce_sum(tf.log(self.x_mean) * self.x + tf.log(1 - self.x_mean) * (1 - self.x), axis=(1, 2, 3))
        self.x_recon = tf.reduce_mean(self.x_recon)

        # Gradient penalty
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_sample
        z_hat = epsilon * self.z_sample + (1 - epsilon) * self.z
        d_hat = discriminator(x_hat, z_hat, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
        ddz = tf.gradients(d_hat, z_hat)[0]
        ddz = tf.sqrt(tf.reduce_sum(tf.square(ddz), axis=1))
        self.d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0) + tf.reduce_mean(tf.square(ddz - 1.0) * 10.0)

        self.d_loss_x = -tf.reduce_mean(self.dx)
        self.d_loss_z = tf.reduce_mean(self.dz)
        self.d_loss = self.d_loss_x + self.d_loss_z + self.d_grad_loss
        self.g_loss = -tf.reduce_mean(self.dz)
        self.i_loss = tf.reduce_mean(self.dx)

        self.d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
        self.g_vars = [var for var in tf.global_variables() if 'g_net' in var.name]
        self.i_vars = [var for var in tf.global_variables() if 'i_net' in var.name]
        self.d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
            self.d_loss, var_list=self.d_vars)
        if 'info' in args.netname:
            self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
                self.g_loss + self.i_loss + 0.2 * self.z_recon + 0.05 * self.x_recon, var_list=self.g_vars + self.i_vars)
        else:
            self.g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
                self.g_loss + self.i_loss, var_list=self.g_vars + self.i_vars)
        # self.i_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(
        #     self.i_loss, var_list=self.i_vars)
        # self.d_gradient_ = tf.gradients(self.d_loss_g, self.g)[0]
        # self.d_gradient = tf.gradients(self.d_logits, self.x)[0]

        self.merged = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('vmi_loss', self.d_loss),
            tf.summary.scalar('vmi_loss', self.i_loss),
            tf.summary.scalar('d_loss_x', self.d_loss_x),
            tf.summary.scalar('d_loss_g', self.d_loss_z),
            tf.summary.scalar('x_recon', self.x_recon),
            tf.summary.scalar('z_recon', self.z_recon),
        ])

        self.ce_ph = tf.placeholder(tf.float32)
        self.norm1_ph = tf.placeholder(tf.float32)
        self.inception_ph = tf.placeholder(tf.float32)
        self.eval_summary = tf.summary.merge([
            tf.summary.scalar('class_ce', self.ce_ph),
            tf.summary.scalar('norm1', self.norm1_ph),
            tf.summary.scalar('inception_score', self.inception_ph)
        ])

        # self.image = tf.summary.image('generated images', self.g, max_images=10)
        self.saver = tf.train.Saver(tf.global_variables())

        self.model_path = "log/%s" % name
        self.fig_path = "%s/fig" % self.model_path
        self.make_model_path()
        self.classifier = Classifier()

    def make_model_path(self):
        if os.path.isdir(self.model_path):
            subprocess.call(('rm -rf %s' % self.model_path).split())
        os.makedirs(self.model_path)
        os.makedirs(self.fig_path)

    def visualize(self, batch_size, sess, save_idx):
        bz = np.random.normal(-1, 1, [batch_size, self.z_dim]).astype(np.float32)
        image = sess.run(self.x_sample, feed_dict={self.z: bz})
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

    def evaluate(self, batch_size, sess):
        data_batches = []
        for i in range(20):
            bz = np.random.normal(-1, 1, [batch_size, self.z_dim]).astype(np.float32)
            image = sess.run(self.x_mean, feed_dict={self.z: bz})
            data_batches.append(image)
        class_dist = self.classifier.class_dist_score(data_batches)
        inception = self.classifier.inception_score(data_batches)
        return class_dist, inception

    def train(self):
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            summary_writer = tf.summary.FileWriter(self.model_path)
            sess.run(tf.global_variables_initializer())
            batch_size = 64

            start_time = time.time()
            for epoch in range(0, 1000):
                batch_idxs = 1093
                for idx in range(0, batch_idxs):
                    if idx % 500 == 0:
                        self.visualize(batch_size, sess, epoch * 2 + idx / 500)
                    if idx % 100 == 0:
                        class_dist, inception = self.evaluate(100, sess)
                        score_summary = sess.run(self.eval_summary, feed_dict={self.ce_ph: class_dist[0],
                                                                               self.norm1_ph: class_dist[1],
                                                                               self.inception_ph: inception})
                        summary_writer.add_summary(score_summary, epoch * batch_idxs + idx)

                    bx = self.dataset.next_batch(batch_size)
                    bz = np.random.normal(-1, 1, [batch_size, self.z_dim]).astype(np.float32)
                    sess.run([self.d_train, self.g_train], feed_dict={self.x: bx, self.z: bz})

                    if idx % 10 == 0:
                        merged, g_loss, d_loss, i_loss = sess.run([self.merged, self.g_loss, self.d_loss, self.i_loss], feed_dict={self.x: bx, self.z: bz})
                        summary_writer.add_summary(merged, epoch * batch_idxs + idx)

                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f, i_loss: %.4f" \
                              % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss, i_loss))

                save_path = "%s/model" % self.model_path
                if os.path.isdir(save_path):
                    subprocess.call(('rm -rf %s' % save_path).split())
                os.makedirs(save_path)
                self.saver.save(sess, save_path, global_step=epoch)


if __name__ == '__main__':

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

