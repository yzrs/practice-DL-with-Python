from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
# 判断电影评论是正面还是负面评论


def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 将评论解码为英文单词 将单词映射为整数索引
# word_index = imdb.get_word_index()
# reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reversed_word_index.get(i - 3, '?') for i in train_data[0]])
# 训练、测试数据向量化
tensor_train_data = vectorize_sequence(train_data)
tensor_test_data = vectorize_sequence(test_data)
# 标签向量化
tensor_train_labels = np.asarray(train_labels).astype('float32')
tensor_test_labels = np.asarray(test_labels).astype('float32')
# 构建网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# 训练集、验证集、测试集
tensor_valid_data = tensor_train_data[:10000]
tensor_partial_train_data = tensor_train_data[10000:]
tensor_valid_labels = tensor_train_labels[:10000]
tensor_partial_train_labels = train_labels[10000:]

my_history = model.fit(tensor_partial_train_data, tensor_partial_train_labels, batch_size=512, epochs=10,
                       validation_data=(tensor_valid_data, tensor_valid_labels), verbose=1)
history_dict = my_history.history
loss_values = history_dict['loss']
valid_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
valid_acc = history_dict['val_accuracy']

fig, axs = plt.subplots(2)

axs[0].plot(loss_values, 'r', label='Training Loss')
axs[0].plot(valid_loss_values, 'b', label='Validation Loss')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(acc,'r',label='Training Acc')
axs[1].plot(valid_acc,'b',label='Validation Acc')
axs[1].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()
plt.subplots_adjust(hspace=0.5)
plt.show()

score = model.evaluate(tensor_test_data, tensor_test_labels, verbose=1,batch_size=512)
print('loss=%s accuracy=%s' % (score[0], score[1]))
res = model.predict(tensor_test_data)
print(res)
