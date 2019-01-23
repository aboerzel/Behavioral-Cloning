import os
from random import randint

import cv2
import numpy as np
from sklearn.utils import shuffle


class DatasetGenerator:
    def __init__(self, images, measurements, batch_size, image_path):

        self.images = np.array(images)
        self.measurements = np.array(measurements)

        self.batch_size = batch_size
        self.image_path = image_path

        self.numImages = len(self.images)
        self.indexes = np.asarray(range(self.numImages))
        shuffle(self.indexes)
        self.batch_index = 0

    def next_batch(self):

        if self.batch_index >= (self.numImages // self.batch_size):
            self.batch_index = 0
            shuffle(self.indexes)

        current_index = self.batch_index * self.batch_size
        batch_indexes = self.indexes[current_index:current_index + self.batch_size]
        self.batch_index += 1
        return self.images[batch_indexes], self.measurements[batch_indexes]

    def read_image(self, filename):
        return cv2.imread(os.path.join(self.image_path, filename), cv2.COLOR_BGR2RGB)

    @staticmethod
    def flip_horizontal(image):
        return cv2.flip(image, 1)

    @staticmethod
    def random_trans(image, steer, trans_range):
        rows, cols, _ = image.shape;
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        steer_ang = steer + tr_x / trans_range * 2 * .2
        tr_y = 40 * np.random.uniform() - 40 / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
        return image_tr, steer_ang

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have reach the desired number of epochs
        while epochs < passes:
            images = []
            steerings = []

            image_names, measurements = self.next_batch()

            for image_name, (steering, throttle, brake, speed) in zip(image_names, measurements):

                image, steering = self.random_trans(self.read_image(image_name), steering, 20)

                # flip about each second image horizontal
                if randint(0, 1) == 1:
                    images.append(self.flip_horizontal(image))
                    steerings.append(-steering)
                else:
                    images.append(image)
                    steerings.append(steering)

            yield shuffle(np.array(images), np.array(steerings))

            # increment the total number of epochs
            epochs += 1
