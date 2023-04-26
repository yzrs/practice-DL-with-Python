# 针对多模态输入的情况  首先要将多个输入标准化
from keras.models import Model
from keras import layers
from keras import Input
from keras import utils
import numpy as np


text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500


text_input = Input(shape=(None,),dtype='int32',name='text')
# 嵌入长度为64的向量
embedded_text = layers.Embedding(text_vocabulary_size,64)(text_input)
# 用LSTM 将长度为64的向量编码为单个向量
encoded_text = layers.LSTM(32)(embedded_text)
question_input = Input(shape=(None,),dtype='int32',name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1)
answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

# 将数据输入到多模态模型
num_samples = 1000
max_length = 100
text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
question = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))
answers = np.random.randint(answer_vocabulary_size,size=(num_samples,))
# one-hot encoding
answers = utils.to_categorical(answers,answer_vocabulary_size)
# 使用输入组成的列表来拟合
model.fit([text,question],answers,epochs=10,batch_size=128)
# 使用输入组成的字典来拟合(只有对输入进行命名之后才能用这种方法)
# model.fit({'text':text,'question':question},answers,epochs=30,batch_size=128)
