from model import cnn_model
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def train():
    # train_img_dir = './test_data/train'
    # test_img_dir = './test_data/test'
    # categories = ['0.5', '1', '2', '2.6', '3', '3.5', '4', '4.5', '5', '5.5', '6.1', '6.5', '7', '7.5', '8', '8.5', '8.9', '9.5', '10', '10.6', '11', '11.4', '12', '12.4', '13', '13.6', '14', '14.5', '15', '15.7', '16.2', '16.5', '17.1', '17.6', '18', '18.5', '18.9', '19.6', '20']

    # image_w = 64
    # image_h = 64

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)

    batch_size=16
    epoch=100

    X_train, X_test, y_train, y_test = np.load('./numpy_data/multi_image_data.npy', allow_pickle=True)
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255
    model = cnn_model()

    model_path = './model/cnn_model_batchsize{0}_epoch{1}_v2.h5'.format(batch_size, epoch)
    # checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='var_loss', patience=6)

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, validation_data=(X_test, y_test))
    model.save(model_path)

    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))

    plt.plot(x_len, y_vloss, marker='.', c='red', label='var_set_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    return X_test, y_test

def evaluate1():
    X_test, y_test = train()
    model = load_model('./model/cnn_model_batchsize16_epoch100_v2.h5')
    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

if __name__ == '__main__':
    evaluate1()    
