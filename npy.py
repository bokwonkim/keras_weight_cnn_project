from model import cnn_model
import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

train_img_dir = './test_data/train'
test_img_dir = './test_data/test'
categories = ['0.5', '1', '2', '2.6', '3', '3.5', '4', '4.5', '5', '5.5', '6.1', '6.5', '7', '7.5', '8', '8.5', '8.9', '9.5', '10', '10.6', '11', '11.4', '12', '12.4', '13', '13.6', '14', '14.5', '15', '15.7', '16.2', '16.5', '17.1', '17.6', '18', '18.5', '18.9', '19.6', '20']
nb_classes = len(categories)

image_w = 64 # or 64
image_h = 64 # or 64

pixels = image_h * image_w * 3

X = []
Y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = train_img_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./numpy_data/multi_image_data_128.npy", xy)