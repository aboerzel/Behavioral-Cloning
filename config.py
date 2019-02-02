# root path of the dataset
DATASET_ROOT_PATH = "../sample_driving_data"

# driving log cvs file
DRIVING_LOG = "driving_log.csv"

# path to the output directory used for models, storing plots, etc.
OUTPUT_PATH = "output"

# network image size
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
IMAGE_DEPTH = 3

# training parameter
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1.0e-4
L2_WEIGHT = 0.001

# number of bins for data distribution
NUM_DATA_BINS = 21

# steering threshold between straight-ahead driving and cornering
STEERING_THRESHOLD = 0.15

# steering correction for left and right camera
STEERING_CORRECTION = 0.25
