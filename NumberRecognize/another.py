from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import utils
# 载入数据集
(trainX, trainY), (testX, testY) = mnist.load_data()
print('Train: X=%s, Y=%s' %(trainX.shape, trainY.shape))
print('Test: X=%s, Y=%s' %(testX.shape, testY.shape))
# 画出一些图
for i in range(16):
    plt.subplot(4,4,1+i)
    plt.imshow(trainX[i])
plt.show()
# trainX.shape[0] 表示样本数量  将图像从RGB转为单通道灰度图
# 灰度值是通过将RGB值加权平均得到的。在计算灰度值时，绿色的权重最高，红色的权重次之，蓝色的权重最低。
# 这是因为人眼对绿色的敏感度最高，对蓝色的敏感度最低。
trainX = trainX.reshape(trainX.shape[0],28,28,1)
testX = testX.reshape(testX.shape[0],28,28,1)
# 如果不进行缩放，那么像素值较大的特征会对模型的训练产生更大的影响，而像素值较小的特征则会被忽略。
trainX = trainX.astype('float32') / 255
testX = testX.astype('float32') / 255
trainY = utils.np_utils.to_categorical(trainY,10)
testY = utils.np_utils.to_categorical(testY,10)
model = models.Sequential()
# 卷积操作会减小输出特征图的大小 如果你想保持输出特征图的大小不变，可以使用padding参数来指定填充方式。
# kernel_size必须是奇数。这是因为卷积核的中心需要对齐到像素点上，
# 而如果kernel_size是偶数，那么中心就会落在两个像素点之间，这样就无法对齐了。
model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25)) # in case of overfitting
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# dropout 防止过拟合
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# epoch指所有的数据训练的轮次(训练多少遍)
model.fit(trainX, trainY, batch_size=32, epochs=5, verbose=1)
score = model.evaluate(testX, testY, verbose=0)
print('loss=%s accuracy=%s' %(score[0],score[1]))
model.save('numRecognize.h5')



