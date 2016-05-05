from __future__ import division

import math
import tensorflow as tf
import numpy as np
import cPickle as pickle
import os
import sys
sys.path.append(os.getcwd())
from src.backend.data_processing.traces2image import IMAGE_SIZE

NUM_CLASSES = 101
WEIGHT_DECAY = 0
LEARNING_RATE = 1e-3
NUM_EPOCHES = 100
BATCH_SIZE = 500


def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.Variable(initial, name=name)
    if WEIGHT_DECAY > 0:
        weight_decay = tf.mul(tf.nn.l2_loss(var), WEIGHT_DECAY)
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(x, keep_prob):
    W_conv1 = weight_variable([5, 5, 1, 32], 0.1, "W_conv1")
    b_conv1 = bias_variable([32], "b_conv1")

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], 0.1, "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024], 0.1, "W_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, NUM_CLASSES], 0.1, "W_fc2")
    b_fc2 = bias_variable([NUM_CLASSES], "b_fc2")

    softmax_linear = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'))


def training(loss):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    return train_step


def evaluation(logits, y_):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def load_model(save_path):
    class Convnet:
        def __init__(self):
            self.x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
            logits = inference(self.x, 1.0)
            self.recognize = tf.argmax(logits, 1)

            saver = tf.train.Saver()
            self.sess = tf.Session()
            saver.restore(self.sess, save_path)

        def predict(self, images):
            """
            images: [N, IMAGE_SIZE * IMAGE_SIZE]
            """
            images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
            return self.sess.run(self.recognize, feed_dict={self.x: images})

    return Convnet()


if __name__ == "__main__":
    with open('data/prepared_data/CROHME.pkl', 'rb') as f:
        CROHME = pickle.load(f)
    train, val, test = CROHME['train'], CROHME['val'], CROHME['test']
    train, val, test = map(lambda x: (x[0].reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1)), x[1]),
                           [train, val, test])

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    logits = inference(x, keep_prob)
    losses = loss(logits, y_)
    train_step = training(losses)
    accuracy = evaluation(logits, y_)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_size = train[0].shape[0]

        for i in xrange(NUM_EPOCHES):
            perm = np.random.permutation(train_size)
            for j in xrange(int(math.ceil(train_size / BATCH_SIZE))):
                idx = perm[j * BATCH_SIZE: min((j + 1) * BATCH_SIZE, train_size)]
                batch = train[0][idx, :], train[1][idx, :]
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

                train_accuracy = sess.run(accuracy, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 0.5})
                print "epoch %d, batch %d, training accuracy %g%%" % (i, j, train_accuracy * 100)

            val_accuracy = sess.run(accuracy, feed_dict={
                x: val[0], y_: val[1], keep_prob: 1.0})
            print "epoch %d, validation accuracy %g%%" % (i, val_accuracy * 100)

        test_accuracy = sess.run(accuracy, feed_dict={
            x: test[0], y_: test[1], keep_prob: 1.0})
        print "test accuracy %g%%" % (test_accuracy * 100)

        save_path = saver.save(sess, "models/convnet/convnet.ckpt")
        print "Model saved in file: %s" % save_path
