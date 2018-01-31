from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# Train on limited data
class LimitedMnist:
    def __init__(self, size, binary=False):
        self.data_ptr = 0
        self.full_mnist = input_data.read_data_sets('mnist_data')
        if size > self.full_mnist.train.images.shape[0]:
            size = self.full_mnist.train.images.shape[0]
        self.size = size
        self.data = self.full_mnist.train.images
        np.random.shuffle(self.data)
        self.data = np.reshape(self.data[:size], [-1, 28, 28, 1])
        if binary:
            self.data = np.rint(self.data)
        self.binary = binary

    def next_batch(self, batch_size):
        assert batch_size <= self.size
        prev_ptr = self.data_ptr
        self.data_ptr += batch_size
        if self.data_ptr > self.size:
            self.data_ptr -= self.size
            return np.concatenate([self.data[prev_ptr:], self.data[:self.data_ptr]], axis=0)
        else:
            return self.data[prev_ptr:self.data_ptr]

    def test_batch(self, batch_size):
        batch_x = self.full_mnist.test.next_batch(batch_size)[0]
        batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
        if self.binary:
            batch_x = np.rint(batch_x)
        return batch_x

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    limited_mnist = LimitedMnist(200, binary=True)
    while True:
        image_x = limited_mnist.test_batch(150)
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.imshow(image_x[i, :, :, 0])
        plt.show()