# 残差连接解决了梯度消失(反向传播的信号向下传播时如果需要经过很多层，可能出现信号非常微弱甚至完全消失)和表示瓶颈(线性模型的表示限制)问题
# 残差连接是让前面某层的输出作为后面某层的输入，从而在序列网络中有效地创造了一条捷径
# 前面层的输出没有与后面层的激活连接在一起，而是与后面层的激活相加
from keras import layers
from keras import Input

x = Input((None,150,150,3))
y = layers.Conv2D(128,3,activation='relu',padding='same')(x)
y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
# 将原始x与输出的特征图相加
y = layers.add([y,x])

# 原始x与输出特征图尺寸不同时需要使用下采样
residual = layers.Conv2D(128,1,strides=2,padding='same')(x)
y = layers.add([y,residual])


