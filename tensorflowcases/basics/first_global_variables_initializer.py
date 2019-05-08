import tensorflow as tf

x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y+y+2

#初始化所有的variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)