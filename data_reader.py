import csv

import numpy as np


def read_samples_from_file(driving_log_filepath):
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

            # skip if speed is 0 - not representative for driving behavior
            if abs(speed) > 0:
                image_paths.append((center, left, right))
                measurements.append((steering, throttle, brake, speed))

    return image_paths, measurements
