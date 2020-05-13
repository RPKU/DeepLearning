import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

x_data = np.float32(np.random.rand(1, 100))
y_data = 0.1 * x_data + 0.3 + np.random.normal(0.0, 0.1)

#
# plt.scatter(x_data, y_data)
# plt.show()

W = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(0, 300):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b), sess.run(loss))
