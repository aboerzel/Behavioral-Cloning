import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
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

print("[INFO] loading data...")
image_names, measurements = read_samples_from_file(os.path.join(data_folder, config.DRIVING_LOG),
                                                   config.STEERING_CORRECTION)

plt.hist(measurements[:, 0], bins=config.NUM_DATA_BINS)
plt.savefig('./examples/steering_distribution_before.png')
plt.show()

image_names, measurements = distribute_data(image_names, measurements)

plt.hist(measurements[:, 0], bins=config.NUM_DATA_BINS)
plt.savefig('./examples/steering_distribution_after.png')
plt.show()

# split into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(image_names, measurements, test_size=0.2, shuffle=True)

print('X_train: {}'.format((len(X_train))))
print('y_train: {}'.format((len(y_train))))
print('X_valid: {}'.format((len(X_valid))))
print('y_valid: {}'.format((len(y_valid))))


def read_image(filename):
    return cv2.imread(os.path.join(data_folder, 'IMG', filename))


def preprocess_image(img):
    # crop region of interest
    new_img = img[50:140, :, :]
    # apply little blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # scale to 66x200x3 (same as nVidia)
    #new_img = cv2.resize(new_img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    #new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    return new_img


def random_brightness(img):
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:, :, 0] + value) > 255
    else:
        mask = (new_img[:, :, 0] + value) < 0
    new_img[:, :, 0] += np.where(mask, 0, value)
    return new_img.astype(np.uint8)


def shift_horizontal(img, angle):
    # Shift image and transform steering angle
    trans_range = 80
    shift_x = trans_range * np.random.uniform() - trans_range / 2
    shift_y = 40 * np.random.uniform() - 40 / 2
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    angle_adj = shift_x / trans_range * 2 * 0.2
    return img, angle + angle_adj


def flip_horizontal(img):
    return cv2.flip(img, 1)


def generate_train_batch(image_names, measurements, batch_size):
    num_images = len(image_names)
    indexes = np.asarray(range(num_images))
    random.shuffle(indexes)
    batch_index = 0

    while True:
        images = []
        steerings = []

        if batch_index >= (num_images // batch_size):
            batch_index = 0
            random.shuffle(indexes)

        current_index = batch_index * batch_size
        batch_indexes = indexes[current_index:current_index + batch_size]
        batch_index += 1

        for image_name, (steering, throttle, brake, speed) in zip(image_names[batch_indexes],
                                                                  measurements[batch_indexes]):
            image = preprocess_image(read_image(image_name))
            image = random_brightness(image)
            #image, steering = shift_horizontal(image, steering)

            images.append(image)
            steerings.append(steering)

            if abs(steering) > config.FLIP_STEERING_THRESHOLD:
                images.append(flip_horizontal(image))
                steerings.append(-steering)

        yield shuffle(np.array(images), np.array(steerings))


def generate_validation_batch(image_names, measurements, batch_size):
    num_images = len(image_names)
    indexes = np.asarray(range(num_images))
    random.shuffle(indexes)
    batch_index = 0

    while True:
        images = []
        steerings = []

        if batch_index >= (num_images // batch_size):
            batch_index = 0
            random.shuffle(indexes)

        current_index = batch_index * batch_size
        batch_indexes = indexes[current_index:current_index + batch_size]
        batch_index += 1

        for image_name, (steering, throttle, brake, speed) in zip(image_names[batch_indexes],
                                                                  measurements[batch_indexes]):
            image = preprocess_image(read_image(image_name))

            images.append(image)
            steerings.append(steering)

            if abs(steering) > config.FLIP_STEERING_THRESHOLD:
                images.append(flip_horizontal(image))
                steerings.append(-steering)

        yield shuffle(np.array(images), np.array(steerings))


train_generator = generate_train_batch(X_train, y_train, args['batch_size'])

val_generator = generate_validation_batch(X_valid, y_valid, args['batch_size'])


class Normalization:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        # normalize and mean centering between -1.0 and +1.0
        model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=input_shape))
        return model


class Nvidia:
    @staticmethod
    def build(base_model):
        base_model.add(Conv2D(24, (5, 5), strides=(2, 2)))
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(36, (5, 5), strides=(2, 2)))
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(48, (5, 5), strides=(2, 2)))
        base_model.add(Activation('elu'))

        base_model.add(Dropout(0.5))

        base_model.add(Conv2D(64, (3, 3)))
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(64, (3, 3)))
        base_model.add(Activation('elu'))

        base_model.add(Dropout(0.5))

        base_model.add(Flatten())

        base_model.add(Dense(100))
        base_model.add(Activation('elu'))

        base_model.add(Dense(50))
        base_model.add(Activation('elu'))

        base_model.add(Dense(10))
        base_model.add(Activation('elu'))

        base_model.add(Dense(1))
        return base_model


def get_callbacks():
    model_filepath = './{}/model.h5'.format(config.OUTPUT_PATH)
    callbacks = [
        #TensorBoard(log_dir="logs".format()),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto', verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, verbose=1),
        #ModelCheckpoint(model_filepath, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, mode='auto',
                          epsilon=0.0001, cooldown=0, min_lr=0, verbose=1)]
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


print("[INFO] create model...")
model = Nvidia.build(Normalization.build(IMAGE_SHAPE))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=args['learning_rate']))

print("[INFO] train model...")
H = model.fit_generator(train_generator,
                        steps_per_epoch=len(X_train) // args['batch_size'],
                        validation_data=val_generator,
                        validation_steps=len(X_valid) // args['batch_size'],
                        epochs=args['epochs'],
                        callbacks=get_callbacks())

plot_and_save_train_history(H)
