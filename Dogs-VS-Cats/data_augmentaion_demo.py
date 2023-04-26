import os
from keras import utils
from keras import preprocessing
import matplotlib.pyplot as plt
dataGen = preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_dir='.\\data\\train'
train_dog_dir = os.path.join(train_dir,'train_dogs')
fNames =[os.path.join(train_dog_dir,fName) for fName in os.listdir(train_dog_dir)]
image_path = fNames[2]
img = utils.image_utils.load_img(path=image_path,target_size=(150,150))
x = utils.image_utils.img_to_array(img)
x = x.reshape((1,)+x.shape)
i=0
for batch in dataGen.flow(x,batch_size=1):
    plt.figure()
    imgPlot = plt.imshow(utils.image_utils.array_to_img(batch[0]))
    i += 1
    if i > 10:
        break
plt.show()
