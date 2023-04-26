from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing import image
from keras import layers
from keras import models
from keras import optimizers
from matplotlib import pyplot as plt


# 使用预训练模型的卷积网络  训练一个新的分类器
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)
base_dir = '.\\data_2'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir, 'test')
dataGen = image.ImageDataGenerator(rescale=1. / 255)
batch_size = 20


# 保存卷积基对directory目录下的sample_count个图片文件的输出结果，作为头部的输入
def extract_feature(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=sample_count)
    generator = dataGen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for images_batch,labels_batch in generator:
        features_batch = conv_base.predict(images_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= sample_count:
            break
        return features,labels


train_count = 8000
validation_count = 2000
test_count = 80

# 获取对应的输出结果
train_features,train_labels = extract_feature(train_dir,train_count)
validation_features,validation_labels = extract_feature(validation_dir,validation_count)
test_features,test_labels = extract_feature(test_dir,test_count)

# (samples,4,4,512) -> (samples,8192) Flatten
train_features = np.reshape(train_features,(train_count,4 * 4 * 512))
validation_features = np.reshape(validation_features,(validation_count,4 * 4 * 512))
test_features = np.reshape(test_features,(test_count,4 * 4 * 512))

# 定义并训练顶部的密集连接器
model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(
    optimizer=optimizers.RMSprop(lr=2e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(validation_features,validation_labels)
)

# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='Training Acc')
plt.plot(epochs,val_acc,'b',label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
