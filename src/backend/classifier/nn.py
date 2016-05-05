from __future__ import division

import math
import tensorflow as tf
from src.backend.data_processing.traces2image import IMAGE_SIZE

NUM_CLASSES = 100
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

BATCH_SIZE = 500


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_network(images, batch_size, number_of_classes):
    """
    :param images: 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    :param batch_size: Batch size
    :param number_of_classes: Number of classes
    :return:
    """
    # conv1
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name='conv1')

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name='conv2')

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='local3')

    # local4
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax, i.e. softmax(WX + b)
    weights = _variable_with_weight_decay('weights', [192, number_of_classes], stddev=1 / 192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [number_of_classes], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


def loss(network, labels):
    return None  # todo


if __name__ == '__main__':
    # todo: get images and labels
    images, labels = None, None

    network = get_network(images, BATCH_SIZE, NUM_CLASSES)

    loss = loss(network, labels)

    # todo: train in batches...
