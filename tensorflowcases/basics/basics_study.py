from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

a = tf.constant(32)
b = tf.constant(10)
c = tf.add(a, b)
sess = tf.Session()
print(sess.run(c))
sess.close()
