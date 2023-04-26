from keras import layers
from keras import Input
from keras import models


vocabulary_size = 50000
num_income_groups = 10

# 一维的卷积神经网络
posts_input = Input(shape=(None,),dtype='int32',name='posts')
embedded_post = layers.Embedding(256,vocabulary_size)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(embedded_post)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128,activation='relu')(x)
# 输出层要有名称
age_prediction = layers.Dense(1,name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1,activation='sigmoid',name='gender')(x)

# 对于不同的输出头，需要为其指定不同的损失函数 并将这些损失合并为单个标量，即全局损失
# 训练使全局损失最小
model = models.Model(posts_input,[age_prediction,income_prediction,gender_prediction])
# 两种不同的写法
# 严重不平衡的损失贡献会导致模型表示针对单个损失值最大的任务优先进行优化，而不考虑其他任务的优化
# 因此，我们需要为其分配不同的权重
# model.compile(optimizer='rmsprop',loss=['mse','categorical_crossentropy','binary_crossentropy']，loss_weights=[0.25,1.,10.])
model.compile(optimizer='rmsprop',
              loss={'age':'mse','income':'categorical_crossentropy','gender':'binary_crossentropy'},
              loss_weights={'age':0.25,'income':1.,'gender':10.})
# posts 训练样本(社交媒体上的发帖转换成的向量)
# age_targets,income_targets,gender_targets是对应的训练labels
# 缺少训练数据
model.fit(posts,[age_targets,income_targets,gender_targets],epochs=10,batch_size=64)



