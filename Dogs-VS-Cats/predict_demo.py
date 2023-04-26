from keras import models
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np


# load model
model = models.load_model('dogs_and_cats.h5')

# predict test data
test_dir = './data/test/t'
test_data_gen = image.ImageDataGenerator(rescale=1./255)
test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
test_images, test_labels = test_generator.next()
# test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)

# plot test images and predictions
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()
# cat 0 dog 1
for i in np.arange(0, 10):
    axs[i].imshow(test_images[i])
    if predictions[i][0] > 0.5:
        axs[i].set_title('Prediction: Dog')
    else:
        axs[i].set_title('Prediction: Cat')
    # axs[i].set_title('Prediction: {:.2f}'.format(predictions[i][0]))
    axs[i].axis('off')
plt.show()
res = model.evaluate(test_generator)
print(res)
