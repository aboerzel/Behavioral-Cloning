import csv
import os

from keras.models import load_model
import numpy as np
from scipy import ndimage

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
car_control_measurements = []
for line in lines:
    filename = line[0].split('/')[-1]
    current_path = os.path.join(data_folder, "IMG", filename)
    image = ndimage.imread(current_path)
    images.append(image)
    # car_control_measurements.append([float(line[3]), float(line[5]), float(line[6])])
    # car_control_measurements.append([float(line[3])])
    car_control_measurements.append(float(line[3]))

model = load_model('output/model.h5')

steering_angle = float(model.predict(np.array([images[0]]), batch_size=1))

print(steering_angle)
