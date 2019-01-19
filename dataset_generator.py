import random
import numpy as np


class DatasetGenerator:
    def __init__(self, images, measurements, image_height, image_width, image_depth, batch_size):

        self.images = images
        self.measurements = measurements

        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.batch_size = batch_size

        self.numImages = self.measurements.shape[0]
        self.indexes = np.asarray(range(self.numImages))
        random.shuffle(self.indexes)
        self.batch_index = 0

    def next_batch(self):

        if self.batch_index >= (self.numImages // self.batch_size):
            self.batch_index = 0
            random.shuffle(self.indexes)

        current_index = self.batch_index * self.batch_size
        batch_indexes = self.indexes[current_index:current_index + self.batch_size]
        self.batch_index += 1
        return self.images[batch_indexes], self.measurements[batch_indexes]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            images = np.empty([self.batch_size, self.image_height, self.image_width, self.image_depth])
            steerings = np.empty(self.batch_size)

            x_data, y_data = self.next_batch()

            for i, ((center, left, right), (steering, throttle, brake, speed)) in enumerate(zip(x_data, y_data)):
                # image = self.augmentor.generate_plate_image(image)
                images[i] = center
                steerings[i] = steering

            yield (images, steerings)

            # increment the total number of epochs
            epochs += 1
