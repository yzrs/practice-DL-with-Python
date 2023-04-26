# 对于给定输入,展示网络中的各个卷积层和池化层输出的特征图
from keras import models
from keras import layers
from keras.preprocessing import image
from keras.utils import image_utils
import numpy as np
from matplotlib import pyplot as plt


model = models.load_model('..\\Dogs-VS-Cats\\dogs_and_cats.h5')
model.summary()
img_path = '..\\Dogs-VS-Cats\\data_2\\test\\cats\\11.jpg'
img = image_utils.load_img(img_path,target_size=(150,150))
img_tensor = image_utils.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255
print(img_tensor.shape)
# plt.imshow(img_tensor[0])
# plt.show()

# 用输入张量和输出张量将模型实例化
# 提取前6层的输出
layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = models.Model(inputs=model.input,outputs=layer_outputs)
activation = activation_model.predict(img_tensor)
# activation里面是6层的输出 每层的输出是(1,height,width,filter_count)
first_layer_act = activation[0]
imgs = np.squeeze(first_layer_act,axis=0)

# 输出中间层的特征图 (第一层有32个通道，即32个特征图)
# for i in range(32):
#     img_show = imgs[:,:,i]
#     plt.imshow(img_show)
#     plt.show()

# 可视化每个中间激活的通道
layer_names = []
for layer in model.layers[:6]:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name,layer_activation in zip(layer_names,activation):
    # 通道数、特征图数
    n_features = layer_activation.shape[-1]
    # feature map(1,size,size,n_features)
    size = layer_activation.shape[1]
    n_rows = n_features // images_per_row
    display_grid = np.zeros((size * n_rows,images_per_row * size))
    for row in range(n_rows):
        for col in range(images_per_row):
            channel_image = layer_activation[0,:,:,row * images_per_row + col]
            # 对特征进行标准化处理
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image,0,255).astype('uint8')
            display_grid[row * size:(row+1)*size,col*size:(col+1)*size] = channel_image
    scale = 1./size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid,aspect='auto',cmap='viridis')
    plt.show()
