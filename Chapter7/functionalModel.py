# 多模态输入 输出
# Inception ResNet
# 函数式API
from keras import Input,layers
from keras.models import Sequential,Model
import numpy as np

# sequential model
seq_model = Sequential()
seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
seq_model.add(layers.Dense(32,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))

# functional model
input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10,activation='softmax')(x)

# keras 会从后台检索从input_tensor到output_tensor中所包含的每一层,将这些层组合成Model
model = Model(input_tensor,output_tensor)
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
x_train = np.random.random((1000,64))
y_train = np.random.random((1000,10))
model.fit(x_train,y_train,epochs=100,batch_size=64)
score = model.evaluate(x_train,y_train)

