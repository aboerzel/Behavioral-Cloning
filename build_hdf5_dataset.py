import argparse
import csv
import os
from hdf5datasetwriter import HDF5DatasetWriter
import config
import cv2
import numpy as np
import progressbar

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--datapath", default=config.DATASET_ROOT_PATH, help="sample driving data path")
ap.add_argument("-s", "--dataset", default=config.HDF5_DATASET_FILENAME, help="hdf5 dataset filename")
ap.add_argument("-i", "--items", default=1000, type=int, help="max images")
args = vars(ap.parse_args())

data_folder = args['datapath']
images_path = os.path.sep.join([data_folder, 'IMG'])
dataset_filepath = os.path.sep.join([data_folder, 'hdf5', args['dataset']])


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


image_paths, measurements = read_samples_from_file(os.path.join(data_folder, config.DRIVING_LOG))

# create HDF5 writer
print("[INFO] building {}...".format(dataset_filepath))
writer = HDF5DatasetWriter((len(image_paths), 3,
                            config.IMAGE_HEIGHT,
                            config.IMAGE_WIDTH,
                            config.IMAGE_DEPTH),
                           (len(image_paths), 4),
                           dataset_filepath)

# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets)
pbar.start()


def read_image(filename):
    return cv2.imread(os.path.join(images_path, filename), cv2.IMREAD_COLOR)


# loop over the image paths
for (i, ((center, left, right), (steering, throttle, brake, speed))) in enumerate(zip(image_paths, measurements)):
    center_img = read_image(center)
    left_img = read_image(left)
    right_img = read_image(right)

    # add the images and measurements # to the HDF5 dataset file
    writer.add([[center_img, left_img, right_img]], [[steering, throttle, brake, speed]])
    pbar.update(i)

# close the HDF5 writer
pbar.finish()
writer.close()
