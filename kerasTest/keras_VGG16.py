# 使用迁移学习的思想，以VGG16作为模板搭建模型，训练识别手写字体
# 引入VGG16模块
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD

from keras.datasets import mnist
#加载OpenCV

import cv2
import h5py as h5py
import numpy as np

# 建立一个模型，其类型是Keras的Model类对象，我们构建的模型会将VGG16顶层去掉，只保留其余的网络
# 结构。这里用include_top = False表明我们迁移除顶层以外的其余网络结构到自己的模型中
# VGG模型对于输入图像数据要求高宽至少为48个像素点，由于硬件配置限制，我们选用48个像素点而不是原来
# VGG16所采用的224个像素点。即使这样仍然需要24GB以上的内存，或者使用数据生成器


#使用本地的模型文件加载VGG16
# path='./[权重文件存放路径]/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# vgg16 = VGG16(weights=path, include_top=True)
model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(48,48,3))
for layer in model_vgg.layers:
    layer.trainable = False
model = Flatten(name='flatten')(model_vgg.output)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation='softmax')(model)
model_vgg_mnist = Model(inputs=model_vgg.input, outputs=model, name='vgg16')

#打印模型结构，包括所需要的参数
model_vgg_mnist.summary()

#新的模型不需要训练原有的卷积结构里面的1471万个参数，但是注意参数还是来自于最后输出层的前两个
#全连接层，一共有1.2亿个参数需要训练
sgd = SGD(lr=0.5, decay=1e-5)
model_vgg_mnist.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

#因为VGG16对网络输入层的要求， 我们用opencv把图像从32*32变成224*224，把黑白变成rgb
#并把训练数据转化为张量形式，供keras输入

# 先读入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data("./test_data_home/mnist.npz")
# 看一下数据集的样子
print(X_train.shape)
print(X_train[0].shape)
print(y_train[0])

#讲灰度图片转换为彩色图片
X_train = [cv2.cvtColor(cv2.resize(i, (48,48)), cv2.COLOR_GRAY2RGB)
           for i in X_train]

#下面concatenate做的事情就是把每个样本按照行堆叠在一起，因为是np下面的方法，
#所以返回的是ndarry
#np.newaxis它本质是None，arr是（48,48,3），arr[None]是(1,48,48,3)
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
X_test = [cv2.cvtColor(cv2.resize(i, (48,48)), cv2.COLOR_GRAY2RGB)
         for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')

print(X_train.shape)
print(X_test.shape)

X_train /= 255
X_test /= 255

def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

model_vgg_mnist.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe),
                    epochs=10, batch_size=100)
# model_vgg_mnist_pretrain.summary()