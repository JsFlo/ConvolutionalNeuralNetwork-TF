import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# used for printing accuracy, sets the dropout to 1 (no droput)
def printAccuracy(accuracy, step, inputPlaceholder, correctLabelPlaceholder, inputs, correctLabels, keep_prob):
    train_accuracy = accuracy.eval(
        feed_dict={inputPlaceholder: inputs, correctLabelPlaceholder: correctLabels, keep_prob: 1.0})
    print('step %d, training accuracy %g' % (step, train_accuracy))


def printShape(tensor):
    print(tensor.shape)


# pooling 2x2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Creates a convolutional layer with a stride of 1
def getHiddenLayer(lastLayer, filterX, filterY, inputChannels, features):
    # the conv layer uses 'SAME" padding to preserve the input dimensions (it's zero-padded)
    convLayer = getConvLayer(lastLayer, filterX, filterY, inputChannels, features)
    # pool 2x2, cut it in half (ex. 28 x 28 => 14 x 14 => 7 x7 ...)
    return max_pool_2x2(convLayer)


# Note padding = 'SAME' will output same dimension as lastLayer
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


# based on the image example (does not use dropout)
def buildExampleModel(x):
    conv1 = getHiddenLayer(x, 5, 5, 1, 32)
    printShape(conv1)
    conv2 = getHiddenLayer(conv1, 3, 3, 32, 64)
    printShape(conv2)
    conv2_flattend = tf.reshape(conv2, [-1, 5 * 5 * 64])
    fc1 = getFullyConnectedLayer(conv2_flattend, 5 * 5 * 64, 1024)
    fc2 = getFullyConnectedLayer(fc1, 1024, 512)
    return getFullyConnectedLayer(fc2, 512, 10)


# expects x to be of shape 28 x 28
def buildModel(x, keep_prob):
    conv1 = getHiddenLayer(x, 2, 2, 1, 32)

    conv2 = getHiddenLayer(conv1, 3, 3, 32, 64)

    conv3 = getHiddenLayer(conv2, 2, 2, 64, 128)
    # flatten the current 3 x 3 with 128 depth into a single row/column
    conv3_flattened = tf.reshape(conv3, [-1, 3 * 3 * 128])

    fullyConnected1 = getFullyConnectedLayer(conv3_flattened, 3 * 3 * 128, 1024)
    fullyConnected1_dropout = tf.nn.dropout(fullyConnected1, keep_prob)

    # fully connected layer 2
    fullyConnected2 = getFullyConnectedLayer(fullyConnected1_dropout, 1024, 512)
    # fully connected 3
    return getFullyConnectedLayer(fullyConnected2, 512, 10)


def main():
    # Load the data from the mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 28 x 28 mnist images = 784 row
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # reshape 784 back to 28 by 28
    # [? , width, height, # color channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 10 hot vectors (0 - 9)
    yCorrectLabels = tf.placeholder(tf.float32, shape=[None, 10])

    # used for dropout later, hold a ref so we can remove it during testing
    keep_prob = tf.placeholder(tf.float32)
    yModel = buildModel(x_image, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yCorrectLabels, logits=yModel))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(yModel, 1), tf.argmax(yCorrectLabels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            batch = mnist.train.next_batch(50)
            if i % 25 == 0:
                printAccuracy(accuracy, i, x, yCorrectLabels, batch[0], batch[1], keep_prob)

            train_step.run(feed_dict={x: batch[0], yCorrectLabels: batch[1], keep_prob: 0.5})

        print(
            'test accuracy %g' % accuracy.eval(
                feed_dict={x: mnist.test.images, yCorrectLabels: mnist.test.labels, keep_prob: 1.0}))


main()
