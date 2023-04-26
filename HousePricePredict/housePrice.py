from keras import layers
from keras import models
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import boston_housing


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    # mse: mean squared error   mae: mean absolute error
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model


def smooth_curve(points,factor=0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous*factor+point*(1-factor))
        else:
            smooth_points.append(point)
    return smooth_points


(train_data,train_target),(test_data,test_target) = boston_housing.load_data()
# 数据取值差异较大时，需要对每个特征进行标准化，对于输入数据中的每个特征减去标准值，再除以标准差
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
# K折验证
k = 4
# / for floating division   // for integer division
num_val_samples = len(train_data) // k
num_epoch = 100
all_scores = []
all_mae_history = []
for i in range(k):
    print('processing fold #', i)
    # 准备第k个分区的数据  验证数据
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_target[i * num_val_samples: (i+1) * num_val_samples]
    # 其他的数据作为训练数据
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_target[:i*num_val_samples],train_target[(i+1)*num_val_samples:]],axis=0)
    my_model = build_model()
    my_history = my_model.fit(partial_train_data,partial_train_targets,epochs=num_epoch,batch_size=32,verbose=1,validation_data=(val_data,val_targets))
    mae_history = my_history.history['val_mae']
    all_mae_history.append(mae_history)
    # model.fit(partial_train_data,partial_train_targets,epochs=num_epoch,batch_size=32,verbose=1)
    # val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
    # all_scores.append(val_mae)
# all_mae_history 里面包含k个列表，每个列表有num_epoch个值
# all_mae_history[i][j]表示第i折的第j个epoch对应的mae  所以每次取一列(第1、2、……、K折)的第j个epoch的均值
average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epoch)]
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation Mean Absolut Error')
plt.show()
# out = np.mean(all_scores)
# print(out)
smooth_mae_history = average_mae_history[10:]
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('epoch')
plt.ylabel('val_mae')
plt.show()
