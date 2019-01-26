import argparse
import os
from random import sample, randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Flatten, Dense, Lambda, Conv2D, Activation, Dropout, BatchNormalization, Cropping2D
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

IMAGE_SHAPE = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_DEPTH)


class Normalization:
    @staticmethod
    def build(input_shape):
        model = Sequential()
        # normalize and mean centering between -0.5 and +0.5
        # model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=input_shape))
        model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=input_shape))
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # remove the sky and the car front
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
        base_model.add(Dropout(0.25))
        base_model.add(Dense(50))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))
        base_model.add(Dropout(0.25))
        base_model.add(Dense(10))
        base_model.add(BatchNormalization())
        base_model.add(Activation('elu'))
        # base_model.add(Dropout(0.25))
        base_model.add(Dense(1))
        return base_model


def get_callbacks():
    model_filepath = './{}/model.h5'.format(config.OUTPUT_PATH)
    callbacks = [
        # TensorBoard(log_dir="logs".format()),
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


# plt.hist(measurements[:, 0], bins=21)
# plt.savefig('./examples/steering_distribution_before.png')
# plt.show()


def distribute_data(image_names, measurements):
    num_hist, idx_hist = np.histogram(measurements[:, 0], 21)
    # max = int(np.median(num_hist))
    max = int(np.average(num_hist))

    # max = 1000

    for i in range(len(num_hist)):
        if num_hist[i] > max:
            # find the index where values fall within the range
            match_idx = np.where((measurements[:, 0] >= idx_hist[i]) & (measurements[:, 0] < idx_hist[i + 1]))[0]
            # randomly choose up to the maximum
            to_be_deleted = sample(list(match_idx), len(match_idx) - max)
            measurements = np.delete(measurements, to_be_deleted, axis=0)
            image_names = np.delete(image_names, to_be_deleted)

    return image_names, measurements


# image_names, measurements = distribute_data(image_names, measurements)

# plt.hist(np.array(measurements)[:, 0], bins=21)
# plt.savefig('./examples/steering_distribution_after.png')
# plt.show()

# split into train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(image_names, measurements, test_size=0.20, shuffle=True)

num_bins = 20
num_hist, idx_hist = np.histogram(y_train[:, 0], num_bins)
data_bins = []

for i in range(num_bins):
    match_idx = np.where((y_train[:, 0] >= idx_hist[i]) & (y_train[:, 0] < idx_hist[i + 1]))[0]
    data_bins.append((X_train[match_idx], y_train[match_idx]))


def read_image(filename):
    return cv2.imread(os.path.join(data_folder, 'IMG', filename))


def preprocess_image(img):
    # crop region of interest
    # new_img = img[50:140, :, :]
    # apply little blur
    # new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    # scale to 66x200x3 (same as nVidia)
    # new_img = cv2.resize(new_img, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return new_img


def random_distort(img, angle):
    new_img = img.astype(float)

    # random brightness - the mask bit keeps values from going beyond (0,255)
    # value = np.random.randint(-28, 28)
    # if value > 0:
    #     mask = (new_img[:, :, 0] + value) > 255
    # if value <= 0:
    #     mask = (new_img[:, :, 0] + value) < 0
    # new_img[:, :, 0] += np.where(mask, 0, value)

    # random shadow - full height, random left/right side, random darkening
    # h, w = new_img.shape[0:2]
    # mid = np.random.randint(0, w)
    # factor = np.random.uniform(0.6, 0.8)
    # if np.random.rand() > .5:
    #     new_img[:, 0:mid, 0] *= factor
    # else:
    #     new_img[:, mid:w, 0] *= factor

    # randomly shift horizon
    h, w, _ = new_img.shape
    horizon = 2 * h / 5
    v_shift = np.random.randint(-h / 8, h / 8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon + v_shift], [w, horizon + v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return new_img.astype(np.uint8), angle


def flip_horizontal(image):
    return cv2.flip(image, 1)


def generate_train_batch(data_bins, batch_size):
    while True:
        images = []
        steerings = []

        bin_indexes = shuffle(range(len(data_bins)))
        n = 0
        while len(images) < batch_size:
            bin_index = bin_indexes[n % len(bin_indexes)]
            n += 1

            (image_names, measurements) = data_bins[bin_index]

            if len(image_names) < 1:
                continue

            sample_ind = randint(0, len(image_names) - 1)
            image_name = image_names[sample_ind]
            steering = measurements[sample_ind][0]
            image = preprocess_image(read_image(image_name))
            image, steering = random_distort(image, steering)

            images.append(image)
            steerings.append(steering)

            # flip about each second image horizontal
            if abs(steering) > 0.35:
                images.append(flip_horizontal(image))
                steerings.append(-steering)

        yield shuffle(np.array(images), np.array(steerings))


def generate_validation_batch(X_data, y_data, batch_size):
    while True:
        images = []
        steerings = []

        shuffle(X_data, y_data)
        n = 0
        while len(images) < batch_size:
            image_name = X_data[n]
            steering = y_data[n][0]
            n += 1

            image = preprocess_image(read_image(image_name))

            images.append(image)
            steerings.append(steering)

            # flip about each second image horizontal
            if abs(steering) > 0.35:
                images.append(flip_horizontal(image))
                steerings.append(-steering)

        yield shuffle(np.array(images), np.array(steerings))


train_generator = generate_train_batch(data_bins, args['batch_size'])
val_generator = generate_validation_batch(X_valid, y_valid, args['batch_size'])

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
