# root paths
DATASET_ROOT_PATH = "../sample_driving_data"

# driving log cvs file
DRIVING_LOG = "driving_log.csv"

# define the path to the output directory used for models, storing plots, classification reports, etc.
OUTPUT_PATH = "output"

# network image size
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 66
IMAGE_DEPTH = 3

# training parameter
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1.0e-4

# number of bins for data distribution
NUM_DATA_BINS = 30

# steering correction for left and right camera
STEERING_CORRECTION = 0.25

# flip images only if steering angle greater than this threshold
FLIP_STEERING_THRESHOLD = 0.3
