import tensorflow as tf

#构造一个tensor, 使得输出一个numpy矩阵[[1 0][0 1]]
mat = tf.constant([[1,0],[0,1]])
sess = tf.Session()
print(sess.run(mat))
#tensor还可以有名字, 在定义每个tensor的时候添加参数name的值就可以.
# 这是一个可选参数, 不过在后面有很大的意义

my_name_is_hello = tf.constant('hello',name='hello')
my_name_is_world = tf.constant('world',name='world')

print('tensor {}:{}'.format(my_name_is_hello.name,sess.run(my_name_is_hello)))
print('tensor {}:{}'.format(my_name_is_world.name,sess.run(my_name_is_world)))
