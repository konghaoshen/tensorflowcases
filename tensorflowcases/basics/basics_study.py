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
'''
上面这段代码看上去非常简单,从结果上来看是在计算一个数学表达式32+10的值,却也完整展示了tensorflow运行的基本构架.
首先,第一行a = tf.constant(32)定义了一个常量tensor,它的值为32,
第二行也是类似.在我们运行tensorflow程序的时候,任何数据都必须转换成tensor类型才能够进入这个系统,
我们先牢记这一点,之后会对它进行优缺点分析.那么现在我们就有了两个常量tensor.
但是仅仅定义了两个用于存储数据的tensor毫无用处,我们希望能够实现的是这两个数的加法运算.
相信大家小时候学数学加减法的时候老师都会在黑板上作出这样的图
'''
