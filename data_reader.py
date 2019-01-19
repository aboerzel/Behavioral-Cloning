import csv

import numpy as np


def read_samples_from_file(driving_log_filepath):
    image_paths = []
    measurements = []
    with open(driving_log_filepath) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            image_paths.append((line[0].split('/')[-1], line[1].split('/')[-1], line[2].split('/')[-1]))
            measurements.append((float(line[3]), float(line[4]), float(line[5]), float(line[6])))
    return np.array(image_paths), np.array(measurements)