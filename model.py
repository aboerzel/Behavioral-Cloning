import argparse
import os
import random
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import config
from data_reader import read_samples_from_file

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--datapath", default=config.DATASET_ROOT_PATH, help="sample driving data path")
ap.add_argument('-l', "--learning_rate", default=config.LEARNING_RATE, type=float, help='learning rate')
ap.add_argument('-b', "--batch_size", default=config.BATCH_SIZE, type=int, help='batch size')
ap.add_argument('-e', "--epochs", default=config.NUM_EPOCHS, type=int, help='max number of epochs')
args = vars(ap.parse_args())

data_folder = args['datapath']

IMAGE_SHAPE = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_DEPTH)


class Preprocessing:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=input_shape))  # normalize between -0.5 and +0.5
        # model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # remove the sky and the car front
        return model


class Nvidia:
    @staticmethod
    def build(base_model):
        base_model.add(Conv2D(24, (5, 5), strides=(2, 2)))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(36, (5, 5), strides=(2, 2)))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(48, (5, 5), strides=(2, 2)))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(64, (3, 3)))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Conv2D(64, (3, 3)))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Dropout(0.5))

        base_model.add(Flatten())

        base_model.add(Dense(100))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Dense(50))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Dense(10))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))

        base_model.add(Dense(1))
        return base_model


def get_callbacks():
    model_filepath = './{}/model.h5'.format(config.OUTPUT_PATH)
    callbacks = [
        TensorBoard(log_dir="logs".format()),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=4, mode='auto', verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=1e-4, cooldown=0,
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


print("[INFO] loading data...")
image_names, measurements = read_samples_from_file(os.path.join(data_folder, config.DRIVING_LOG),
                                                   config.STEERING_CORRECTION)

measurements = np.array(measurements)


# plt.hist(np.array(measurements)[:, 0], bins=21)
# plt.savefig('./examples/steering_distribution_before.png')
# plt.show()


def distribute_data(image_names, measurements, min=500, max=750):
    image_names = np.array(image_names)
    measurements = np.asarray(measurements)

    # create histogram to know what needs to be added
    steering_angles = measurements[:, 0]

    num_hist, idx_hist = np.histogram(steering_angles, 21)

    for i in range(1, len(num_hist)):
        if num_hist[i - 1] < min:
            # find the index where values fall within the range
            match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

            if len(match_idx) == 0:
                continue

            count_to_be_added = min - num_hist[i - 1]
            while len(match_idx) < count_to_be_added:
                match_idx = np.concatenate((match_idx, match_idx))

            # randomly choose up to the minimum
            to_be_added = np.random.choice(match_idx, count_to_be_added)
            measurements = np.concatenate((measurements, measurements[to_be_added]))
            image_names = np.concatenate((image_names, image_names[to_be_added]))
            steering_angles = np.concatenate((steering_angles, measurements[to_be_added][:, 0]))

        elif num_hist[i - 1] > max:
            # find the index where values fall within the range
            match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

            while len(match_idx) > max:
                # randomly choose up to the maximum
                to_be_deleted = np.random.choice(match_idx, len(match_idx) - max)
                measurements = np.delete(measurements, to_be_deleted, axis=0)
                image_names = np.delete(image_names, to_be_deleted)
                steering_angles = np.delete(steering_angles, to_be_deleted)
                match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

    return image_names, measurements


# image_names, measurements = distribute_data(image_names, measurements)

# plt.hist(np.array(measurements)[:, 0], bins=21)
# plt.savefig('./examples/steering_distribution_after.png')
# plt.show()

# split into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(image_names, measurements, test_size=0.20, shuffle=True)


def random_trans(image, steer, trans_range):
    rows, cols, _ = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return image_tr, steer_ang


def read_image(filename):
    return cv2.imread(os.path.join(data_folder, 'IMG', filename), cv2.COLOR_BGR2RGB)


def make_roi(image):
    crop_img = image[50:140, 0:320, :]
    crop_img = cv2.resize(crop_img, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), cv2.INTER_AREA)
    return crop_img


def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


def flip_horizontal(image):
    return cv2.flip(image, 1)


def generate_train_batch(image_names, measurements, batch_size, train_mode):
    while True:
        images = []
        steerings = []

        image_names, measurements = shuffle(image_names, measurements)
        rand_indexes = np.random.choice(np.arange(len(image_names)), batch_size)

        for rand_index in rand_indexes:
            image = make_roi(random_brightness(read_image(image_names[rand_index])))
            steering = measurements[rand_index]

            if train_mode:
                image, steering = random_trans(image, steering, 20)

            images.append(image)
            steerings.append(steering)

            # flip about each second image horizontal
            # if randint(0, 1) == 1 and abs(steering) > 0.15:
            #     images.append(flip_horizontal(image))
            #     steerings.append(-steering)
            # else:
            #     images.append(image)
            #     steerings.append(steering)

        yield np.array(images), np.array(steerings)


train_generator = generate_train_batch(X_train, y_train[:, 0], args['batch_size'], True)
val_generator = generate_train_batch(X_valid, y_valid[:, 0], args['batch_size'], False)

print("[INFO] create model...")
model = Nvidia.build(Preprocessing.build(IMAGE_SHAPE))
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
