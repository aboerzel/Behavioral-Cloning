import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Dropout, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import config
from data import read_samples_from_file, distribute_data

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--datapath", default=config.DATASET_ROOT_PATH, help="sample driving data path")
ap.add_argument('-l', "--learning_rate", default=config.LEARNING_RATE, type=float, help='learning rate')
ap.add_argument('-b', "--batch_size", default=config.BATCH_SIZE, type=int, help='batch size')
ap.add_argument('-e', "--epochs", default=config.NUM_EPOCHS, type=int, help='max number of epochs')
args = vars(ap.parse_args())

data_folder = args['datapath']

IMAGE_SHAPE = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH)

# load data
print("[INFO] loading data...")
image_paths, measurements = read_samples_from_file(os.path.join(data_folder, config.DRIVING_LOG),
                                                   config.STEERING_CORRECTION)

# ensure even distribution of the steering angles
X_train, y_train = distribute_data(image_paths, measurements)

# split into train and validation data and shuffle data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, shuffle=True)

# print the length of the train- and validation data
print('X_train: {}'.format((len(X_train))))
print('y_train: {}'.format((len(y_train))))
print('X_valid: {}'.format((len(X_valid))))
print('y_valid: {}'.format((len(y_valid))))


# read image from dataset
def read_image(filename):
    return cv2.imread(os.path.join(data_folder, filename))


# image augmentation methods:
# apply random brightness to the image to simulate sunlight, darkness, shadows, ect.
def random_brightness(img):
    # Convert 2 HSV colorspace from BGR colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    # Convert to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


# shift the image randomly horizontal and vertical to simulate a wobble of the camera
def random_shift(img, angle):
    # random shift image and transform steering angle
    trans_range = 80
    shift_x = trans_range * np.random.uniform() - trans_range / 2
    shift_y = 40 * np.random.uniform() - 40 / 2
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    new_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    delta_angle = shift_x / trans_range * 2 * 0.2
    return new_img, angle + delta_angle


# flip the image horizontally to simulate the mirror situation
def flip_horizontal(img, angle):
    return cv2.flip(img, 1), -angle


# train generator, uses image augmentation and horizontal flip
def generate_train_batch(image_paths, measurements, batch_size):
    while True:
        image_paths, measurements = shuffle(image_paths, measurements)

        for i in range(0, len(image_paths), batch_size):

            batch_images = []
            batch_steerings = []

            batch_image_paths = image_paths[i: i + batch_size]
            batch_measurements = measurements[i: i + batch_size]

            for image_name, (steering, throttle, brake, speed) in zip(batch_image_paths, batch_measurements):
                image = read_image(image_name)
                image = random_brightness(image)
                image, steering = random_shift(image, steering)

                if random.randint(0, 1) == 1:
                    image, steering = flip_horizontal(image, steering)

                batch_images.append(image)
                batch_steerings.append(steering)

            yield shuffle(np.array(batch_images), np.array(batch_steerings))


# validation generator, uses only horizontal flip, but no further image augmentation
def generate_validation_batch(image_paths, measurements, batch_size):
    while True:
        image_paths, measurements = shuffle(image_paths, measurements)

        for i in range(0, len(image_paths), batch_size):

            batch_images = []
            batch_steerings = []

            batch_image_paths = image_paths[i: i + batch_size]
            batch_measurements = measurements[i: i + batch_size]

            for image_name, (steering, throttle, brake, speed) in zip(batch_image_paths, batch_measurements):
                image = read_image(image_name)

                if random.randint(0, 1) == 1:
                    image, steering = flip_horizontal(image, steering)

                    batch_images.append(image)
                    batch_steerings.append(steering)

            yield shuffle(np.array(batch_images), np.array(batch_steerings))


# construct the train- and validation generators
train_generator = generate_train_batch(X_train, y_train, args['batch_size'])
val_generator = generate_validation_batch(X_valid, y_valid, args['batch_size'])


# base model for image preprocessing.
# crops the sky and the car front and normalizes the image data between -0.5 and +0.5
class Preprocessing:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        # normalize and mean centering between -0.5 and +0.5
        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
        # remove the sky and the car front
        model.add(Cropping2D(cropping=((60, 20), (0, 0))))
        return model


# nVidia model, derives from the base model
class Nvidia:
    @staticmethod
    def build(base_model):
        base_model.add(Conv2D(24, (5, 5), strides=(2, 2), kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Conv2D(64, (3, 3), kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Flatten())
        base_model.add(Dropout(0.5))
        base_model.add(Dense(100, kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Dense(50, kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Dense(10, kernel_regularizer=l2(config.L2_WEIGHT)))
        base_model.add(Activation('elu'))
        base_model.add(Dense(1))
        return base_model


# define callbacks
# - stop if there is no improvement during training
# - save best model ever trained
# - auto reduce learning rate if there is no improvement
def get_callbacks():
    model_filepath = './{}/model.h5'.format(config.OUTPUT_PATH)
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, verbose=1)]
    return callbacks


# plot and save the training history
def plot_and_save_train_history(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
    plt.title("Mean Squared Error Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Mean squared error loss")
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig('./{}/training-history.png'.format(config.OUTPUT_PATH))
    plt.show()
    cv2.waitKey(0)


# construct model
print("[INFO] create model...")
model = Nvidia.build(Preprocessing.build(IMAGE_SHAPE))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=args['learning_rate']))

# start training
print("[INFO] train model...")
H = model.fit_generator(train_generator,
                        steps_per_epoch=len(X_train) // args['batch_size'],
                        validation_data=val_generator,
                        validation_steps=len(X_valid) // args['batch_size'],
                        epochs=args['epochs'],
                        callbacks=get_callbacks(),
                        verbose=1)
# plot training history
plot_and_save_train_history(H)
