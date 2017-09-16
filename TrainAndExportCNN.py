import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

EXPORT_DIR = './model'

weights = {
    # 2x2 conv, 1 input, 64 outputs
    'wc1': tf.Variable(tf.random_normal([2, 2, 1, 64])),
    # 3x3 conv, 64 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    # 5 x 5 conv, 64 input, 10 output (through stride = 2)
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 10])),
    # fully connected, 11 * 11 * 10 inputs, 1024 outputs
    'wf1': tf.Variable(tf.random_normal([11 * 11 * 10, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([10])),
    'bf1': tf.Variable(tf.random_normal([1024])),
    'bout': tf.Variable(tf.random_normal([10]))
}


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Note this does not add zero-padding (padding = 'VALID') (padding = 'SAME' will output same dimension as lastLayer)
def getConvLayer(lastLayer, weight, bias, stride=1):
    conv1 = tf.nn.conv2d(lastLayer, weight, strides=[1, stride, stride, 1], padding='VALID')
    return tf.nn.relu(conv1 + bias)


# fully connected with relu
def getFullyConnectedLayer(lastLayer, input, output, bias):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.relu(tf.matmul(lastLayer, W_fc1) + b_fc1)


def getFullyConnectedLayer(lastLayer, inputOutputWeight, bias):
    return tf.nn.relu(tf.matmul(lastLayer, inputOutputWeight) + bias)


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

conv1 = getConvLayer(x_image, weights['wc1'], biases['bc1'])  # 28 x 28 x 1 => 27 x 27 x 64
printShape(conv1)
conv2 = getConvLayer(conv1, weights['wc2'], biases['bc2'])  # 27 x 27 x 64 => 25 x 25 x 64
printShape(conv2)
# conv with stride of 2 to reduce size (instead of pooling)
conv3 = getConvLayer(conv2, weights['wc3'], biases['bc3'], 2)  # 25 x 25 x 64 => 11 x 11 x 10
printShape(conv3)

# flatten conv3 to connect to the fully connected layer next
conv3_flat = tf.reshape(conv3, [-1, 11 * 11 * 10])  # 11 x 11 x 10 => 1210
printShape(conv3_flat)

# Fully Connected Layer 1
fully_connected1 = getFullyConnectedLayer(conv3_flat, weights['wf1'], biases['bf1'])  # 1210 => 1024
printShape(fully_connected1)

# used for dropout later, hold a ref so we can remove it during testing
keep_prob = tf.placeholder(tf.float32)
fully_connected_drop1 = tf.nn.dropout(fully_connected1, keep_prob)
print("Dropout")

# fully connected layer 2
fully_connected2 = getFullyConnectedLayer(fully_connected_drop1, weights['out'], biases['bout'])  # 1024 => 10
printShape(fully_connected2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yCorrectLabels, logits=fully_connected2))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(fully_connected2, 1), tf.argmax(yCorrectLabels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
        batch = mnist.train.next_batch(50)
        if i % 25 == 0:
            printAccuracy(accuracy, i, x, yCorrectLabels, batch[0], batch[1], keep_prob)

        train_step.run(feed_dict={x: batch[0], yCorrectLabels: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(
        feed_dict={x: mnist.test.images, yCorrectLabels: mnist.test.labels, keep_prob: 1.0}))

    WC1 = weights['wc1'].eval(sess)
    BC1 = biases['bc1'].eval(sess)

    WC2 = weights['wc2'].eval(sess)
    BC2 = biases['bc2'].eval(sess)

    WC3 = weights['wc3'].eval(sess)
    BC3 = biases['bc3'].eval(sess)

    WF1 = weights['wf1'].eval(sess)
    BF1 = biases['bf1'].eval(sess)

    W_OUT = weights['out'].eval(sess)
    B_OUT = biases['bout'].eval(sess)

# Create new graph for exporting
g = tf.Graph()
with g.as_default():
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")

    WC1 = tf.constant(WC1, name="WC1")
    BC1 = tf.constant(BC1, name="BC1")
    x_image = tf.reshape(x_2, [-1, 28, 28, 1])
    CONV1 = getConvLayer(x_image, WC1, BC1)

    WC2 = tf.constant(WC2, name="WC2")
    BC2 = tf.constant(BC2, name="BC2")
    CONV2 = getConvLayer(conv1, WC2, BC2)

    WC3 = tf.constant(WC3, name="WC3")
    BC3 = tf.constant(BC3, name="BC3")
    CONV3 = getConvLayer(conv1, WC3, BC3)

    CONV3_FLAT = tf.reshape(CONV3, [-1, 11 * 11 * 10])

    WF1 = tf.constant(WF1, name="WF1")
    BF1 = tf.constant(BF1, name="BF1")
    FC1 = getFullyConnectedLayer(CONV3_FLAT, WF1, BF1)

    W_OUT = tf.constant(W_OUT, name="W_OUT")
    B_OUT = tf.constant(B_OUT, name="B_OUT")

    OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    graph_def = g.as_graph_def()
    tf.train.write_graph(graph_def, EXPORT_DIR, 'model_graph.pb', as_text=False)