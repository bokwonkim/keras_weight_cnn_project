from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

data_aug_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.5, zoom_range=[0.5, 2.0], horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
img = load_img('./test_data/train/20/20(1).jpg')

x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='./test_data/train/20/', save_format='jpg'):
    i += 1
    if i > 1000:
        break