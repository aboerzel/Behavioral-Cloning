import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import config
from dataset_generator import DatasetGenerator
from hdf5datasetloader import Hdf5DatasetLoader

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--datapath", default=config.DATASET_ROOT_PATH, help="sample driving data path")
ap.add_argument("-s", "--dataset", default=config.HDF5_DATASET_FILENAME, help="hdf5 dataset filename")
ap.add_argument('-l', "--learning_rate", default=config.LEARNING_RATE, type=float, help='learning rate')
args = vars(ap.parse_args())

data_folder = args['datapath']
dataset_filepath = os.path.sep.join([data_folder, 'hdf5', args['dataset']])

IMAGE_SHAPE = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH)


class Preprocessing:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # remove the sky and the car front
        return model


class Nvidia:
    @staticmethod
    def build(base_model):
        base_model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
        base_model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
        base_model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
        base_model.add(Conv2D(64, (3, 3), activation='elu'))
        base_model.add(Conv2D(64, (3, 3), activation='elu'))
        base_model.add(Dropout(0.5))
        base_model.add(Flatten())
        base_model.add(Dense(100, activation='elu'))
        base_model.add(Dense(50, activation='elu'))
        base_model.add(Dense(10, activation='elu'))
        base_model.add(Dense(1))
        return base_model


def get_callbacks():
    model_filepath = './{}/model.h5'.format(config.OUTPUT_PATH)
    callbacks = [
        TensorBoard(log_dir="logs".format()),
        EarlyStopping(monitor='loss', min_delta=0, patience=5, mode='auto', verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=1e-4, cooldown=0,
                          min_lr=0)]
    return callbacks


def plot_and_save_train_history(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
    plt.title("Mean Squared Error Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Mean squared error loss")
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.savefig('./{}/training-history.png'.format(config.OUTPUT_PATH))
    plt.show()
    cv2.waitKey(0)


def combine_images(images, measurements, correction):
    new_images = []
    new_images.extend(images[:, 0])  # center
    new_images.extend(images[:, 1])  # left
    new_images.extend(images[:, 2])  # right

    steering = measurements[:, 0]
    throttle = measurements[:, 1]
    brake = measurements[:, 2]
    speed = measurements[:, 3]

    new_measurements = []
    new_measurements.extend(measurements)
    new_measurements.extend(np.array([[x + correction for x in steering], throttle, brake, speed]).T)
    new_measurements.extend(np.array([[x - correction for x in steering], throttle, brake, speed]).T)

    return np.array(new_images), np.array(new_measurements)


print("[INFO] loading data...")
loader = Hdf5DatasetLoader()
images, measurements = loader.load(dataset_filepath)
images, measurements = combine_images(images, measurements, 0.2)

X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=0.20)

print("[INFO] create model...")
model = Nvidia.build(Preprocessing.build(IMAGE_SHAPE))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=args['learning_rate']))

trainGen = DatasetGenerator(X_train, y_train, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH,
                            config.BATCH_SIZE)

valGen = DatasetGenerator(X_valid, y_valid, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH,
                          config.BATCH_SIZE)

print("[INFO] train model...")
H = model.fit_generator(trainGen.generator(),
                        steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
                        validation_data=valGen.generator(),
                        validation_steps=valGen.numImages // config.BATCH_SIZE,
                        epochs=config.NUM_EPOCHS,
                        callbacks=get_callbacks())

plot_and_save_train_history(H)
