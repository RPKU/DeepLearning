import tensorflow as tf

w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])

y = tf.matmul(w, x)

print(y.numpy())

