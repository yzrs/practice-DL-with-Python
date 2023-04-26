from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
# train_images = train_images.reshape(train_images.shape[0],28,28,1)
train_images = train_images.reshape((60000,28 * 28))
train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape(test_images.shape[0],28,28,1)
test_images = test_images.reshape((10000,28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

network = models.Sequential()
# network.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
# network.add(layers.Flatten())
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(train_images,train_labels,epochs=5,batch_size=128)
test_loss,test_acc = network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)


