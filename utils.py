import tensorflow as tf
import numpy.random as npr
import numpy as np


def sample_Z(m, n, mode='uniform'):
    if mode == 'uniform':
        return npr.uniform(-1., 1., size=[m, n])
    if mode == 'gaussian':
        return np.clip(npr.normal(0, 0.1, (m, n)), -1, 1)


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def lrelu(input, leak=0.2, scope='lrelu'):
    return tf.maximum(input, leak * input)


if __name__ == '__main__':
    computeTSNE()
