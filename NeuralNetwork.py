import tensorflow.compat.v1 as tf


class NeuralNetwork:

    def __init__(self, numHiddenUnits=50, trainingIteration=10000, batchSize=32, learning_rate=0.1):
        tf.disable_v2_behavior()
        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.trainingIteration = trainingIteration
        self.numHiddenUnites = numHiddenUnits
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

        x = tf.placeholder(tf.float32, shape=[None, self.inputSize])
        y = tf.placeholder(tf.float32, shape=[None, self.numClasses])

        w1 = tf.Variable(tf.truncated_normal([self.inputSize, self.numHiddenUnites], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1), [self.numHiddenUnites])
        w2 = tf.Variable(tf.truncated_normal([self.numHiddenUnites, self.numClasses], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1), [self.numClasses])

        hiddenLayerOutput = tf.matmul(x, w1) + b1
        hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
        finalOutput = tf.matmul(hiddenLayerOutput, w2) + b2
        finalOutput = tf.nn.relu(finalOutput)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        accuracy, update_op = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.argmax(finalOutput, 1))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer, feed_dict={data: x_input, label: y_input})

        for i in range(self.trainingIteration):
            x_batch, y_batch = sess.run(data_element)
            _, trainingLoss, op = sess.run([opt, loss, update_op], feed_dict={x: x_batch, y: y_batch})
            if i % 100 == 0:
                print(op)
        print("train finish")

        x_batch, y_batch = sess.run(data_element)
        self.w1, self.b1, self.w2, self.b2 = sess.run([w1, b1, w2, b2], feed_dict={x: x_batch, y: y_batch})
        sess.close()

    def predict(self, data, label):
        x = tf.constant(data, tf.float32)

        hiddenLayerOutput = tf.matmul(x, self.w1) + self.b1
        hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
        finalOutput = tf.matmul(hiddenLayerOutput, self.w2) + self.b2
        finalOutput = tf.nn.relu(finalOutput)

        accuracy, update_op = tf.metrics.accuracy(labels=tf.argmax(label, 1), predictions=tf.argmax(finalOutput, 1))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("The accuracy: %(a)f" % {"a": sess.run(update_op)})
