import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import Tensor


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Note this does not add zero-padding (padding = 'VALID') (padding = 'SAME' will output same dimension as lastLayer)
def getConvLayer(lastLayer, filterX, filterY, inputChannels, features, stride=1):
    filter1 = weight_variable([filterX, filterY, inputChannels, features])
    conv1 = tf.nn.conv2d(lastLayer, filter1, strides=[1, stride, stride, 1], padding='VALID')
    bias1 = bias_variable([features])
    return tf.nn.relu(conv1 + bias1)


# fully connected with relu
def getFullyConnectedLayer(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.relu(tf.matmul(lastLayer, W_fc1) + b_fc1)

# used for printing accuracy, sets the dropout to 1 (no droput)
def printAccuracy(accuracy, step, inputPlaceholder, correctLabelPlaceholder, inputs, correctLabels, keep_prob):
    train_accuracy = accuracy.eval(
        feed_dict={inputPlaceholder: inputs, correctLabelPlaceholder: correctLabels, keep_prob: 1.0})
    print('step %d, training accuracy %g' % (step, train_accuracy))

def printShape(tensor):
    print(tensor.shape)


# Load the data from the mnist
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 28 x 28 mnist images = 784 row
x = tf.placeholder(tf.float32, shape=[None, 784])

# reshape 784 back to 28 by 28
# [? , width, height, # color channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 10 hot vectors (0 - 9)
yCorrectLabels = tf.placeholder(tf.float32, shape=[None, 10])

conv1 = getConvLayer(x_image, 2, 2, 1, 64) # 28 x 28 x 1 => 27 x 27 x 64
printShape(conv1)
conv2 = getConvLayer(conv1, 3, 3, 64, 64) # 27 x 27 x 64 => 25 x 25 x 64
printShape(conv2)
# conv with stride of 2 to reduce size (instead of pooling)
conv3 = getConvLayer(conv2, 5, 5, 64, 10, 2) # 25 x 25 x 64 => 11 x 11 x 10
printShape(conv3)

# flatten conv3 to connect to the fully connected layer next
conv3_flat = tf.reshape(conv3, [-1, 11 * 11 * 10]) # 11 x 11 x 10 => 1210
printShape(conv3_flat)

# Fully Connected Layer 1
fully_connected1 = getFullyConnectedLayer(conv3_flat, 11 * 11 * 10, 100) # 1210 => 100
printShape(fully_connected1)

# used for dropout later, hold a ref so we can remove it during testing
keep_prob = tf.placeholder(tf.float32)
fully_connected_drop1 = tf.nn.dropout(fully_connected1, keep_prob)
print("Dropout")

# fully connected layer 2
fully_connected2 = getFullyConnectedLayer(fully_connected_drop1, 100, 10) # 1210 => 100
printShape(fully_connected2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yCorrectLabels, logits=fully_connected2))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(fully_connected2, 1), tf.argmax(yCorrectLabels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            printAccuracy(accuracy, i, x, yCorrectLabels, batch[0], batch[1], keep_prob)

        train_step.run(feed_dict={x: batch[0], yCorrectLabels: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(
        feed_dict={x: mnist.test.images, yCorrectLabels: mnist.test.labels, keep_prob: 1.0}))




