import csv
import os

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, BatchNormalization, Dropout, MaxPooling2D, Conv2D

data_folder = '../sample_driving_data'
driving_log = 'driving_log.csv'

lines = []
with open(os.path.join(data_folder, driving_log)) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

# print(lines)

images = []
measurements = []
for line in lines:
    filename = line[0].split('/')[-1]
    current_path = os.path.join(data_folder, "IMG", filename)
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append([float(line[3]), float(line[4]), float(line[5]), float(line[6])])

measurements = np.array(measurements)


# print(measurements)

class MiniVGGNet:
    @staticmethod
    def build(num_classes):
        model = Sequential()

        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

        # Layer 1
        # Conv Layer 1 => 28x28x6
        model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 3)))

        # Layer 2
        # Pooling Layer 1 => 14x14x6
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        # Conv Layer 2 => 10x10x16
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14, 14, 6)))

        # Layer 4
        # Pooling Layer 2 => 5x5x16
        model.add(MaxPooling2D(pool_size=2, strides=2))

        # Flatten
        model.add(Flatten())

        # Layer 5
        # Fully connected layer 1 => 120x1
        model.add(Dense(units=120, activation='relu'))

        model.add(Dropout(0.5))

        # Layer 6
        # Fully connected layer 2 => 84x1
        model.add(Dense(units=84, activation='relu'))

        model.add(Dropout(0.5))

        # Output Layer => num_classes x 1
        model.add(Dense(units=num_classes))

        # show and return the constructed network architecture
        model.summary()
        return model


batch_size = 32
epochs = 3

X_train = np.array(images)
y_train = np.array(measurements[:, 0])

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))

model = MiniVGGNet.build(1)

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=batch_size, epochs=epochs)

model.save('output/model.h5')
