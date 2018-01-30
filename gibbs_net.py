import tensorflow as tf
from dataset import *
from abstract_network import *
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def make_model_path(name):
    log_path = os.path.join('log/gibbs', name)
    if os.path.isdir(log_path):
        subprocess.call(('rm -rf %s' % log_path).split())
    os.makedirs(log_path)
    return log_path

log_path = make_model_path('default_all')


def gibbs_inference(x_list, name='gibbs_net', reuse=False):
    assert len(x_list) != 0
    with tf.variable_scope(name) as vs:
        if reuse:
            vs.reuse_variables()
        res_list1, res_list2 = [], []
        z_list = []
        conv_shape = None
        for x in x_list:
            conv = conv2d_bn_lrelu(x, 32, 4, 2)
            conv = conv2d_bn_lrelu(conv, 32, 4, 1)
            res_list1.append(conv)
            conv = conv2d_bn_lrelu(conv, 64, 4, 2)
            conv = conv2d_bn_lrelu(conv, 64, 4, 1)
            res_list2.append(conv)
            conv_shape = conv.get_shape().as_list()[1:]
            conv = tf.reshape(conv, [-1, np.prod(conv.get_shape().as_list()[1:])])
            fc = fc_lrelu(conv, 512)
            z_list.append(fc)
        noise = tf.random_normal(tf.stack([tf.shape(z_list[0])[0], 10]))
        z = tf.concat([noise] + z_list, axis=-1)
        fc = fc_relu(z, int(np.prod(conv_shape)))
        conv = tf.reshape(fc, [-1] + conv_shape)
        conv = tf.concat([conv] + res_list2, axis=-1)
        conv = conv2d_t_bn_relu(conv, 64, 4, 1)
        conv = conv2d_t_bn_relu(conv, 32, 4, 2)
        noise = tf.random_normal(tf.stack([tf.shape(conv)[0], tf.shape(conv)[1], tf.shape(conv)[2], 10]))
        conv = tf.concat([conv, noise] + res_list1, axis=-1)
        conv = conv2d_t_bn_relu(conv, 32, 4, 1)
        output = tf.contrib.layers.convolution2d_transpose(conv, 1, 4, 2, activation_fn=tf.sigmoid)
        return output


def discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        conv1 = conv2d_lrelu(x, 64, 4, 2)
        conv2 = conv2d_lrelu(conv1, 128, 4, 2)
        conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(conv2, 1024)
        fc2 = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
        return fc2


x_in = tf.placeholder(tf.float32, [None, 28, 28, 1])
x_ref = tf.placeholder(tf.float32, [None, 28, 28, 1])
width = 16
padw = 28 - width
slice_size = [-1, width, width, -1]
lu = tf.slice(x_in, [0, 0, 0, 0], slice_size)
ru = tf.slice(x_in, [0, padw, 0, 0], slice_size)
lb = tf.slice(x_in, [0, 0, padw, 0], slice_size)
rb = tf.slice(x_in, [0, padw, padw, 0], slice_size)

for step in range(3):
    lu = gibbs_inference([ru, lb, rb], name='gibbs_lu', reuse=(step != 0))
    ru = gibbs_inference([lu, rb, lb], name='gibbs_ru', reuse=(step != 0))
    lb = gibbs_inference([lu, rb, ru], name='gibbs_lb', reuse=(step != 0))
    rb = gibbs_inference([lb, ru, lu], name='gibbs_rb', reuse=(step != 0))

# Final Step
lu = tf.pad(lu, [[0, 0], [0, padw], [0, padw], [0, 0]])
ru = tf.pad(ru, [[0, 0], [padw, 0], [0, padw], [0, 0]])
lb = tf.pad(lb, [[0, 0], [0, padw], [padw, 0], [0, 0]])
rb = tf.pad(rb, [[0, 0], [padw, 0], [padw, 0], [0, 0]])

lu_mask = tf.pad(tf.ones([width, width, 1], tf.float32), [[0, padw], [0, padw], [0, 0]])
ru_mask = tf.pad(tf.ones([width, width, 1], tf.float32), [[padw, 0], [0, padw], [0, 0]])
lb_mask = tf.pad(tf.ones([width, width, 1], tf.float32), [[0, padw], [padw, 0], [0, 0]])
rb_mask = tf.pad(tf.ones([width, width, 1], tf.float32), [[padw, 0], [padw, 0], [0, 0]])

x_gen = ((lu * (1 - ru_mask) + ru) * (1 - lb_mask) + lb) * (1 - rb_mask) + rb


d = discriminator(x_ref)
d_ = discriminator(x_gen, reuse=True)

# Variational mutual information maximization
# Gradient penalty
epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = epsilon * x_gen + (1 - epsilon) * x_ref
d_hat = discriminator(x_hat, reuse=True)

ddx = tf.gradients(d_hat, x_hat)[0]
ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=(1, 2, 3)))
d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

d_loss_x = -tf.reduce_mean(d)
d_loss_g = tf.reduce_mean(d_)
confusion = d_loss_x + d_loss_g
d_loss = d_loss_x + d_loss_g + d_grad_loss
g_loss = -tf.reduce_mean(d_)
loss = d_loss + g_loss

d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
g_vars = [var for var in tf.global_variables() if 'gibbs' in var.name]
d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(d_loss, var_list=d_vars)
g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9).minimize(g_loss, var_list=g_vars)

train_summary = tf.summary.merge([
    tf.summary.scalar('g_loss', g_loss),
    tf.summary.scalar('d_loss_x', d_loss_x),
    tf.summary.scalar('d_loss_g', d_loss_g),
    tf.summary.scalar('confusion', confusion),
    tf.summary.scalar('gradient', d_grad_loss)
])

red_lu = tf.pad(lu, [[0, 0], [0, 0], [0, 0], [0, 2]])
green_ru = tf.pad(ru, [[0, 0], [0, 0], [0, 0], [1, 1]])
blue_lb = tf.pad(lb, [[0, 0], [0, 0], [0, 0], [2, 0]])
white_rb = tf.tile(rb, [1, 1, 1, 3])
colored = ((red_lu * (1 - ru_mask) + green_ru) * (1 - lb_mask) + blue_lb) * (1 - rb_mask) + white_rb

sample_summary = tf.summary.merge([
    create_display(tf.slice(x_gen, [0, 0, 0, 0], [64, -1, -1, -1]), 'samples'),
    create_display(tf.slice(colored, [0, 0, 0, 0], [64, -1, -1, -1]), 'colored'),
    create_multi_display([
        tf.slice(lu, [0, 0, 0, 0], [64, -1, -1, -1]),
        tf.slice(ru, [0, 0, 0, 0], [64, -1, -1, -1]),
        tf.slice(lb, [0, 0, 0, 0], [64, -1, -1, -1]),
        tf.slice(rb, [0, 0, 0, 0], [64, -1, -1, -1])
    ], 'patches')
])


sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)

dataset = MnistDataset()

batch_size = 100

start_time = time.time()
for idx in range(0, 10000000):
    bx_ref = dataset.next_batch(batch_size)
    bx_in = np.random.uniform(0.0, 1.0, [batch_size, 28, 28, 1]).astype(np.float32)
    sess.run([d_train, g_train], feed_dict={x_ref: bx_ref, x_in: bx_in})

    if idx % 10 == 0:
        summary_val = sess.run(train_summary, feed_dict={x_ref: bx_ref, x_in: bx_in})
        summary_writer.add_summary(summary_val, idx)

    if idx % 500 == 0:
        summary_val = sess.run(sample_summary, feed_dict={x_in: bx_in})
        summary_writer.add_summary(summary_val, idx)
        print("Iteration %d, time=%.1f" % (idx, time.time() - start_time))