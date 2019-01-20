import tensorflow as tf

# input1 = tf.placeholder(tf.float32, [2, 2])
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

# 用placeholder 相当于 等到sesion的时候再给他值
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
