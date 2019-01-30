import csv

import numpy as np
from sklearn.utils import shuffle

import config


def read_samples_from_file(driving_log_filepath, steering_correction):
    image_paths = []
    measurements = []
    with open(driving_log_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip headers
        for line in reader:
            # camera images
            center = line[0].strip()
            left = line[1].strip()
            right = line[2].strip()

            # measurements
            steering = float(line[3])
            throttle = float(line[4])
            brake = float(line[5])
            speed = float(line[6])

            # skip if speed is less than 0.1 because it's not representative for driving behavior
            if abs(speed) < 0.1:
               continue

            if steering > config.STEERING_THRESHOLD:
                image_paths.extend([center, left])
                measurements.extend([(steering, throttle, brake, speed),
                                     (steering + steering_correction, throttle, brake, speed)])
            if steering < -config.STEERING_THRESHOLD:
                image_paths.extend([center, right])
                measurements.extend([(steering, throttle, brake, speed),
                                     (steering - steering_correction, throttle, brake, speed)])
            else:
                image_paths.append(center)
                measurements.append((steering, throttle, brake, speed))

    return np.array(image_paths), np.array(measurements)


def distribute_data(image_paths, measurements):
    num_hist, idx_hist = np.histogram(measurements[:, 0], config.NUM_DATA_BINS)

    max_count = int(max(num_hist) * 0.75)

    for i in range(len(num_hist)):
        if num_hist[i] < max_count:
            # find the index where values fall within the range
            match_idx = np.where((measurements[:, 0] >= idx_hist[i]) & (measurements[:, 0] < idx_hist[i + 1]))[0]
            if len(match_idx) < 1:
                continue
            # randomly choose up to the max_count
            to_be_added = np.random.choice(match_idx, max_count - num_hist[i])
            image_paths = np.append(image_paths, image_paths[to_be_added])
            measurements = np.vstack((measurements, measurements[to_be_added]))

    return shuffle(image_paths, measurements)
