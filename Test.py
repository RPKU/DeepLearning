import tensorflow.compat.v1 as tf
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train, x_test = x_train.reshape(-1, 28, 28, 1) / 255.0, x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.one_hot(y_train, 10).numpy()
y_test = tf.one_hot(y_test, 10).numpy()
model = ConvolutionalNeuralNetwork(trainingIteration=1000)
model.train(x_train, y_train)
model.predict(x_test, y_test)
