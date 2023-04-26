import keras
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.datasets import mnist


def load_image(filename):
    img = Image.open(filename).convert('L')
    img = img.resize((28, 28))
    data = np.array(img)
    data = data.reshape(1, 28, 28, 1)
    data = data.astype('float32') / 255
    return data


pic = load_image('number.PNG')
my_model = load_model('numRecognize.h5')
my_model.summary()
res = my_model.predict(pic)
print(res)
