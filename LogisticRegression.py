from typing import Any, Union, Tuple

import tensorflow.compat.v1 as tf


class LogisticRegression:

    def __init__(self, trainingIteration=50000, batchSize=32, learning_rate=0.05):
        tf.disable_v2_behavior()
        self.learning_rate = learning_rate
        self.trainingIterations = trainingIteration
        self.batchSize = batchSize
        self.numClasses = 0
        self.inputSize = 0

    def train(self, x_input, y_input):
        self.numClasses = y_input.shape[1]
        self.inputSize = x_input.shape[1]
        self.__train(x_input, y_input)

    def __train(self, x_input, y_input):
        data = tf.placeholder(tf.float32, shape=[None, self.inputSize])
        label = tf.placeholder(tf.float32, shape=[None, self.numClasses])
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.shuffle(20).batch(self.batchSize).repeat()
        iterator = dataset.make_initializable_iterator()
        data_element = iterator.get_next()

        X = tf.placeholder(tf.float32, shape=[None, self.inputSize])
        y = tf.placeholder(tf.float32, shape=[None, self.numClasses])

        W = tf.Variable(tf.random_normal([self.inputSize, self.numClasses], stddev=0.1))
        b = tf.Variable(tf.constant(0.1), [self.numClasses])

        y_predict = tf.nn.softmax(tf.matmul(X, W) + b)

        loss = tf.reduce_mean(tf.square(y - y_predict))
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        accuracy, update_op = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(y_predict, 1))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer, feed_dict={data: x_input, label: y_input})

        for i in range(self.trainingIterations):
            x_batch, y_batch = sess.run(data_element)
            _, trainingLoss, op = sess.run([opt, loss, update_op], feed_dict={X: x_batch, y: y_batch})
            if i % 1000 == 0:
                print(op)
        print("train finish")

        x_batch, y_batch = sess.run(data_element)
        res_W, res_b = sess.run([W, b], feed_dict={X: x_batch, y: y_batch})
        self.W, self.b = res_W, res_b
        sess.close()

    def predict(self, data, label):
        X = tf.constant(data, tf.float32)
        y_predict = tf.nn.softmax(tf.matmul(X, self.W) + self.b)

        accuracy, update_op = tf.metrics.accuracy(labels=tf.argmax(label, 1), predictions=tf.argmax(y_predict, 1))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("The accuracy: %(a)f" % {"a": sess.run(update_op)})