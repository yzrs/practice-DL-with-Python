import os
import shutil
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# build model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])

#image preprocess
train_dir = './data/train'
test_dir = './data/test'
train_data_gen = image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# 测试集的数据不需要增强
test_data_gen = image.ImageDataGenerator(rescale=1./255)
# image generator
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary',
    subset='training'
)
validation_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary',
    subset='validation'
)
test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

my_his = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50
)
model.save('dogs_and_cats.h5')
his = my_his.history
acc = his['acc']
val_acc = his['val_acc']
loss= his['loss']
val_loss = his['val_loss']
epoch = range(1,len(acc)+1)
plt.plot(epoch,acc,'r.-',label='Training Acc')
plt.plot(epoch,val_acc,'b.-',label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epoch,loss,'r.-',label='Training Loss')
plt.plot(epoch,val_loss,'b.-',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# res = model.evaluate(test_generator,verbose=1)
# print(res)
