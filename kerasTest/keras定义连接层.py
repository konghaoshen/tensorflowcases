#序列模型
#序列模型属于通用模型一种，因为很常见，所以这里单独列出来进行介绍，这种模型各层之间
#是依次顺序执行的关系，在第k层和k+1层之间可以加上各种元素来构建神经网络
#这些元素可以通过一个列表来指定，然后作为参考传递给序列模型来生成相应的模型

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Activation

#Dense相当于构建一个全连接层，32指的是全连接层上面神经元的个数
layers = [Dense(32, input_shape=(784,)),
          Activation('relu'),
          Dense(10),
          Activation('softmax')]
model = Sequential(layers)
model.summary()


model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()