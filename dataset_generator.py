import os
from random import random

import cv2
import numpy as np
from sklearn.utils import shuffle


class DatasetGenerator:
    def __init__(self, images, measurements, image_height, image_width, image_depth, batch_size, steering_correction,
                 image_path):

        self.images = np.array(images)
        self.measurements = np.array(measurements)

        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.batch_size = batch_size
        self.steering_correction = steering_correction
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

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # steering correction for left or right camera image
        steering_correction = 0.20

        # keep looping infinitely -- the model will stop once we have reach the desired number of epochs
        while epochs < passes:
            images = []
            steerings = []

            x_data, y_data = self.next_batch()

            for (center, left, right), (steering, throttle, brake, speed) in zip(x_data, y_data):

                center_image = self.read_image(center)
                images.append(center_image)
                steerings.append(steering)

                images.append(self.read_image(left))
                steerings.append(steering - steering_correction)

                images.append(self.read_image(right))
                steerings.append(steering + steering_correction)

                # flip about each second image horizontal
                if random() < .5:
                    images.append(self.flip_horizontal(center_image))
                    steerings.append(-steering)

            yield shuffle(np.array(images), np.array(steerings))

            # increment the total number of epochs
            epochs += 1
