# 输入空白图像 利用梯度下降让某个过滤器的响应最大化
# 使用随机梯度下降调节输入图像 使得激活最大化
# 使用梯度下降算法来调整输入图像的像素值。
# 由于我们的目标是最大化响应，因此我们需要将损失函数定义为负值。梯度上升->损失变大
from keras.applications import VGG16
from keras import backend as bac
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# 将张量转换为有效图像
def deprocess_image(x):
    # 标准化
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1
    # 将x裁切到(0,1)区间
    x += 0.5
    x = np.clip(x, 0, 1)
    # 将x转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 输入一个层的名称和一个filter索引,返回一个有效的图像张量,表示将特定filter的激活最大化的模式
def generate_pattern(model, layer_name, filter_index, size=150):
    # 计算损失相对于输入图像的梯度
    layer_output = model.get_layer(layer_name).output
    loss = bac.mean(layer_output[:, :, :, filter_index])
    grads = bac.gradients(loss, model.input)[0]
    # 将梯度除以L2范数来标准化(张量中所有值的平方的平均值的平方根)  1e-5是为了避免除以0
    grads /= bac.sqrt(bac.mean(bac.square(grads))) + 1e-5
    # 返回输入图像的损失和梯度
    iterate = bac.function([model.input], [loss, grads])
    # 通过梯度上升让损失最大化
    # 从一张带有噪声的灰度图像开始
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128
    step = 1
    # 运行40次梯度上升 沿着让损失最大化的方向调节输入图像的像素值 得到(1，150，150，3)的浮点数张量
    for i in range(40):
        loss_val, grads_val = iterate([input_img_data])
        input_img_data += grads_val * step
    img = input_img_data[0]
    return deprocess_image(img)


def test1():
    model = VGG16(weights='imagenet', include_top=False)
    layer_name = 'block3_conv1'
    filter_index = 0
    plt.imshow(generate_pattern(model, layer_name, filter_index))
    plt.show()


def test2():
    model = VGG16(weights='imagenet', include_top=False)
    layer_name = 'block1_conv1'
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(model,layer_name,i * 8 + j,size=size)
            horizontal_start = j * size + j * margin
            horizontal_end = horizontal_start + size
            vertical_start = i * size + i * margin
            vertical_end = vertical_start + size
            results[vertical_start:vertical_end,horizontal_start:horizontal_end,:] = filter_img
    plt.figure(figsize=(20,20))
    plt.imshow(deprocess_image(results))
    plt.show()


if __name__ == '__main__':
    model = VGG16(weights='imagenet', include_top=False)
    model.summary()
    test1()
