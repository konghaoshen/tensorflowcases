import tensorflow as tf

x = tf.Variable(3,name='x')
y = tf.Variable(4,name='y')
f = x*x*y+y+2

init = tf.global_variables_initializer()

#interacitvesession 和常规的session不同在于，自动默认设置它自己的为默认的session
#即无需放在with块中，但是这样需要自己来  closs session
#使用与jupyter中 效果比较好
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()