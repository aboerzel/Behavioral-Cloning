import csv
import os
import argparse
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, BatchNormalization, Dropout, MaxPooling2D, Conv2D, \
    Cropping2D
from scipy import ndimage
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", default='../sample_driving_data', help="data folder")
args = vars(ap.parse_args())

data_folder = args['data']
driving_log = 'driving_log.csv'


def read_samples_from_file(driving_log_filepath):
    images = []
    measurements = []
    with open(driving_log_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            images.append([line[0].split('/')[-1], line[1].split('/')[-1], line[2].split('/')[-1]])
            measurements.append(float(line[3]))
    return images, measurements


class Preprocessing:
    @staticmethod
    def build():
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))
        return model


class LeNet:
    @staticmethod
    def build(model):
        model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14, 14, 6)))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        return model


class Nvidia:
    @staticmethod
    def build(model):
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Conv2D(64, (3, 3), activation='elu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        return model


class ModelArchitecture(Enum):
    LeNet = 1
    NVidia = 2


def get_model(model_architecture):
    model = {
        ModelArchitecture.LeNet: LeNet.build(Preprocessing.build()),
        ModelArchitecture.NVidia: Nvidia.build(Preprocessing.build()),
    }[model_architecture]
    print('{} - Model'.format(model_architecture.name))
    model.summary()
    return model


def get_callbacks(model_architecture):
    model_filepath = './output/{}_model.h5'.format(model_architecture)
    callbacks = [
        TensorBoard(log_dir="logs/{}".format(model_architecture)),
        EarlyStopping(monitor='loss', min_delta=0, patience=5, mode='auto', verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=False, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=1e-4, cooldown=0,
                          min_lr=0)]
    return callbacks


def plot_and_save_train_history(H, model_architecture):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
    plt.title("Mean Squared Error Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Mean squared error loss")
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.savefig('./output/training-history_{}.png'.format(model_architecture))
    plt.show()


X, y = read_samples_from_file(os.path.join(data_folder, driving_log))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# os.path.join(data_folder, "IMG", line[0].split('/')[-1]),

# trainig hyperparameter
batch_size = 128
epochs = 1
model_architecture = ModelArchitecture.NVidia

model = get_model(model_architecture)

model.compile(loss='mse', optimizer='adam')

H = model.fit(X_train, y_train,
              validation_split=0.2,
              shuffle=True,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=get_callbacks(model_architecture.name))

plot_and_save_train_history(H, model_architecture.name)
