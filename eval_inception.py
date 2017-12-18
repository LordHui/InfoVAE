import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from abstract_network import *
import tensorflow as tf
from scipy.stats import entropy



def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


class Classifier:
    def __init__(self, load_network=False):
        # Import data
        data_path = 'data'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.mnist = input_data.read_data_sets(data_path, one_hot=True)

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y, self.keep_prob = self.network(self.x)
        y_ = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.y_prob = tf.nn.softmax(self.y)

        train_step = tf.train.AdamOptimizer(0.0002).minimize(cross_entropy)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
        saver = tf.train.Saver([var for var in tf.global_variables() if 'classifier' in var.name])
        self.sess.run(tf.global_variables_initializer())
        if len(glob.glob(os.path.join(data_path, 'classifier.ckpt') + '*')) != 0:
            saver.restore(self.sess, os.path.join(data_path, 'classifier.ckpt'))
            print("Classification model restored, test acc is %.4f" %
                  self.sess.run(accuracy, feed_dict={self.x: np.reshape(self.mnist.test.images, [-1, 28, 28, 1]),
                                                     y_: self.mnist.test.labels, self.keep_prob: 1.0}))
            return
        else:
            print("Classification model reinitialized")

        # Train
        for i in range(10000):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
            self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 0.5})
            if i % 1000 == 0:
                print("Iteration %d, acc=%.4f" % (i,
                      self.sess.run(accuracy, feed_dict={self.x: np.reshape(self.mnist.test.images, [-1, 28, 28, 1]),
                                                         y_: self.mnist.test.labels, self.keep_prob: 1.0})))

        saver.save(self.sess, os.path.join(data_path, 'classifier.ckpt'))

    def class_dist_score(self, data_batches):
        labels = []
        for batch_xs in data_batches:
            batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
            label = self.sess.run(self.y, feed_dict={self.x: batch_xs, self.keep_prob: 1.0})
            label = np.argmax(label, 1)
            labels.append(label)
        label = np.concatenate(labels)
        count = np.bincount(label) / float(label.size)
        ce_score = np.sum(-np.log(count) / float(count.size)) - math.log(count.size)
        norm1_score = np.sum(np.abs(count - 0.1))
        norm2_score = np.sqrt(np.sum(np.square(count - 0.1)))
        # print(count, ce_score, norm1_score, norm2_score)
        return ce_score, norm1_score, norm2_score

    def inception_score(self, data_batches):
        probs = []
        for batch_xs in data_batches:
            batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
            y_prob = self.sess.run(self.y_prob, feed_dict={self.x: batch_xs, self.keep_prob: 1.0})
            probs.append(y_prob)
        y_prob = np.concatenate(probs, axis=0)
        y_marginal = np.mean(y_prob, axis=0)
        # print(y_marginal)
        cond_ent = np.mean([entropy(y_prob[i]) for i in range(y_prob.shape[0])])
        ent = entropy(y_marginal)
        # print(ent, cond_ent)
        return ent - cond_ent

    def network(self, x, reuse=False):
        with tf.variable_scope('classifier') as vs:
            if reuse:
                vs.reuse_variables()
            conv1 = conv2d_bn_lrelu(x, 64, 4, 2)
            conv2 = conv2d_bn_lrelu(conv1, 64, 4, 1)
            conv3 = conv2d_bn_lrelu(conv2, 128, 4, 2)
            conv4 = conv2d_bn_lrelu(conv3, 128, 4, 1)
            conv4 = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
            fc1 = fc_bn_lrelu(conv4, 1024)
            keep_prob = tf.placeholder(tf.float32)
            fc1_drop = tf.nn.dropout(fc1, keep_prob)
            fc2 = tf.contrib.layers.fully_connected(fc1_drop, 10, activation_fn=tf.identity)
            return fc2, keep_prob

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    classifier = Classifier()
    data_batches = []
    for i in range(50):
        data_batches.append(np.random.normal(size=[100, 28, 28, 1]))
    print(classifier.class_dist_score(data_batches))
    print(classifier.inception_score(data_batches))

    data_batches = []
    for i in range(50):
        data_batches.append(np.reshape(classifier.mnist.test.next_batch(100), [100, 28, 28, 1]))
    print(classifier.class_dist_score(data_batches))
    print(classifier.inception_score(data_batches))