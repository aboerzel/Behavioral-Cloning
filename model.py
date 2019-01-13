import csv
import os

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

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
#print(measurements)

batch_size = 32
epochs = 10

X_train = np.array(images)
y_train = np.array(measurements[:, 0])

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=batch_size, epochs=epochs)

model.save('output/model.h5')
