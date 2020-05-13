import tensorflow.compat.v1 as tf


def conv2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x, w, h):
    return tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, 2, 2, 1], padding="SAME")


class ConvolutionalNeuralNetwork:

    def __init__(self, trainingIteration=50000, batchSize=32, pool_size=2,
                 conv_size=5, factor_num=32, keep_prob=0.5):
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        self.keep_prob = keep_prob
        self.batchSize = batchSize
        self.trainingIteration = trainingIteration
        self.height = 0
        self.width = 0
        self.channel = 0
        self.numClasses = 0
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.factor_num = factor_num

    def train(self, data, label):
        self.height = data.shape[1]
        self.width = data.shape[2]
        self.numClasses = label.shape[1]
        self.channel = data.shape[3]
        self.__train(data, label)

    def __train(self, x_input, y_input):

        data = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel])
        label = tf.placeholder(tf.float32, shape=[None, self.numClasses])
        dataset = tf.data.Dataset.from_tensor_slices((data, label))
        dataset = dataset.shuffle(20).batch(self.batchSize).repeat()
        iterator = dataset.make_initializable_iterator()
        data_element = iterator.get_next()

        w_convolution_1 = tf.Variable(
            tf.truncated_normal([self.conv_size, self.conv_size, self.channel, self.factor_num], stddev=0.1))
        b_convolution_1 = tf.Variable(tf.constant(0.1, shape=[self.factor_num]))

        self.x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel])
        self.y = tf.placeholder(tf.float32, shape=[None, self.numClasses])

        h_convolution_1 = conv2d(self.x, w_convolution_1) + b_convolution_1
        h_convolution_1 = tf.nn.relu(h_convolution_1)
        h_pool_1 = max_pool(h_convolution_1, self.pool_size, self.pool_size)

        factor_num_2 = self.factor_num * self.pool_size
        w_convolution_2 = tf.Variable(
            tf.truncated_normal([self.conv_size, self.conv_size, self.factor_num, factor_num_2], stddev=0.1))
        b_convolution_2 = tf.Variable(tf.constant(0.1, shape=[factor_num_2]))
        h_convolution_2 = tf.nn.relu(conv2d(h_pool_1, w_convolution_2) + b_convolution_2)
        h_pool_2 = max_pool(h_convolution_2, self.pool_size, self.pool_size)

        new_size = h_pool_2.shape[1] * h_pool_2.shape[2] * factor_num_2
        w_full_connect_1 = tf.Variable(tf.truncated_normal([new_size, 1024], stddev=0.1))
        b_full_connect_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        h_pool_2_flat = tf.reshape(h_pool_2, [-1, new_size])
        h_full_connect_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, w_full_connect_1) + b_full_connect_1)

        h_full_connect_1_drop = tf.nn.dropout(h_full_connect_1, self.keep_prob)

        w_full_connect_2 = tf.Variable(tf.truncated_normal([1024, self.numClasses], stddev=0.1))
        b_full_connect_2 = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))

        y_predict = tf.matmul(h_full_connect_1_drop, w_full_connect_2) + b_full_connect_2

        crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_predict))
        trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)

        accuracy, self.update_op = tf.metrics.accuracy(labels=tf.argmax(self.y, 1), predictions=tf.argmax(y_predict, 1))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(iterator.initializer, feed_dict={data: x_input, label: y_input})

        for i in range(self.trainingIteration):
            x_batch, y_batch = self.sess.run(data_element)
            op, _ = self.sess.run([self.update_op, trainStep], feed_dict={self.x: x_batch, self.y: y_batch})
            if i % 10 == 0:
                print("Train step: %(s)d, accuracy: %(a)f" % {"s": i, "a": op})
        print("train finish")

        self.w_convolution_1 = w_convolution_1
        self.w_convolution_2 = w_convolution_2
        self.w_full_connect_1 = w_full_connect_1
        self.w_full_connect_2 = w_full_connect_2
        self.b_convolution_1 = b_convolution_1
        self.b_convolution_2 = b_convolution_2
        self.b_full_connect_1 = b_full_connect_1
        self.b_full_connect_2 = b_full_connect_2

    def predict(self, data, label):

        op = self.sess.run(self.update_op, feed_dict={self.x: data, self.y: label})

        print("The accuracy: %(a)f" % {"a": op})