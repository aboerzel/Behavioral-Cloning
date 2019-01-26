import csv
import numpy as np
from random import sample


def read_samples_from_file(driving_log_filepath, steering_correction):
    image_paths = []
    measurements = []
    with open(driving_log_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip headers
        for line in reader:
            # camera images
            center = line[0].split('/')[-1]
            left = line[1].split('/')[-1]
            right = line[2].split('/')[-1]

            # measurements
            steering = float(line[3])
            throttle = float(line[4])
            brake = float(line[5])
            speed = float(line[6])

            # skip if speed is less than 0.1 because it's not representative for driving behavior
            if abs(speed) < 0.1:
                continue

            image_paths.extend([center, left, right])
            measurements.extend([(steering, throttle, brake, speed),
                                 (steering - steering_correction, throttle, brake, speed),
                                 (steering + steering_correction, throttle, brake, speed)])

    return np.array(image_paths), np.array(measurements)


def distribute_data(image_names, measurements):
    num_hist, idx_hist = np.histogram(measurements[:, 0], 21)
    # max = int(np.median(num_hist))
    max = int(np.average(num_hist))

    for i in range(len(num_hist)):
        if num_hist[i] > max:
            # find the index where values fall within the range
            match_idx = np.where((measurements[:, 0] >= idx_hist[i]) & (measurements[:, 0] < idx_hist[i + 1]))[0]
            # randomly choose up to the maximum
            to_be_deleted = sample(list(match_idx), len(match_idx) - max)
            measurements = np.delete(measurements, to_be_deleted, axis=0)
            image_names = np.delete(image_names, to_be_deleted)

    return image_names, measurements
