from keras import models
from keras import layers
import numpy as np
from keras.datasets import reuters
import matplotlib.pyplot as plt
# 判断新闻属于哪一个类别 共46个类别


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def vectorize_sequences_2(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            results[i, index] = 1
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


# data preprocess
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
# one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
# one_hot_test_labels = to_categorical(test_labels)
# model construct
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
fitRes = model.fit(partial_x_train, partial_y_train, epochs=30, batch_size=128, validation_data=(x_val, y_val))
# plot
history_dict = fitRes.history
loss = history_dict['loss']
acc = history_dict['accuracy']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_accuracy']

fig, axs = plt.subplots(2,1)
axs[0].plot(loss,'r.-',label='Training Loss')
axs[0].plot(val_loss,'b.-',label='Validation Loss')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss Value')
axs[0].legend()
axs[1].plot(acc,'r.-',label='Training Acc')
axs[1].plot(val_acc,'b.-',label='Validation Acc')
axs[1].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
plt.subplots_adjust(hspace=0.5)
plt.show()
score = model.evaluate(x_test, one_hot_test_labels, verbose=0)
print('loss=%s accuracy=%s' %(score[0],score[1]))

# model.save('NewsOfMultiClass.h5')
