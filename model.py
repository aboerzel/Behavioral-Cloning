import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, BatchNormalization, Dropout, MaxPooling2D, Conv2D
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default='./output/model.h5', help="model file")
# ap.add_argument("-v", "--video", help="input video")
args = vars(ap.parse_args())

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
    #car_control_measurements.append([float(line[3])])
    car_control_measurements.append(float(line[3]))

car_control_measurements = np.array(car_control_measurements)


# print(measurements)

class LeNet:
    @staticmethod
    def build(num_outputs):
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
        model.add(Dense(units=num_outputs))

        # show and return the constructed network architecture
        model.summary()
        return model


def get_callbacks():
    callbacks = [
        TensorBoard(log_dir="logs/{}".format('Behavioral-Cloning')),
        EarlyStopping(monitor='loss', min_delta=0, patience=5, mode='auto', verbose=1),
        ModelCheckpoint(args['model'], save_best_only=False, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=1e-4, cooldown=0,
                          min_lr=0)]
    return callbacks


def plot_and_save_train_history(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, len(H.history["acc"])), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, len(H.history["val_acc"])), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('./output/training-loss-and-accuracy.png')
    plt.show()


# trainig hyperparameter
batch_size = 128
epochs = 1000

X_train = np.array(images)
y_train = np.array(car_control_measurements)

# model = LeNet.build(car_control_measurements.shape[1])
model = LeNet.build(1)

model.compile(loss='mse', optimizer='adam')

H = model.fit(X_train, y_train,
              validation_split=0.2,
              shuffle=True,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=get_callbacks())

#plot_and_save_train_history(H)
