# **Behavioral Cloning** 

## Udacity Self Driving Car Engineer Nanodegree - Project 4

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_architecture]: ./examples/model-architecture.png "Model Architecture"
[nvidia_model]: ./examples/nVidia_model.png "NVIDIA Model Architecture"
[distribution_1]: ./examples/steering_distribution_before.png "Steering Distribution Before"
[distribution_2]: ./examples/steering_distribution_after.png "Steering Distribution After"
[sample_image]: ./examples/sample_image1.png "Sample Image"
[cropped_image]: ./examples/cropped_image.png "Sample Image"
[random_brightness]: ./examples/random_brightness.png "Random Brightness"
[random_shift]: ./examples/random_shift.png "Random Shift"
[horizontal_flip]: ./examples/horizontal_flip.png "Horizontal Flip"
[train_image_batch]: ./examples/train_image_batch.png "Train Image Batch"
[train_history]: ./examples/training-history.png "Training History"

[video_track_1]: output/version-1/video.mp4 "Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [data.py](data.py) containing the script to load and distribute the data
* [config.py](config.py) defines global parameters for the project
* [drive.py](drive.py) for driving the car in autonomous mode
* `model.h5` the trained convolution neural network
* [video.mp4](output/version-1/video.mp4) video of the recordings by drive.py. while the car is driving track-1 in the simulator.
* `README.md` summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The [data.py](data.py) file contains the code to load the simulator data from the `driving_log.csv` file and to compensate the distribution of the data.

The [config.py](config.py) is used to define some project wide parameters, such as batch size, number of epochs, ect.

### Model Architecture and Training Strategy

#### Model Architectur

My model architecture is bases on the NVIDIA model ([End to End Learning for Self-Driving Cars Paper](https://arxiv.org/pdf/1604.07316v1.pdf)), 
because it has a fairly simple architecture and because it demonstrably provides good results for use with self-driving cars. 
It is a feed-foward layered architecture in which the output of a layer is fed into the overlying layer. 
The network consists of a normalization layer, 5 convolutional layers, a dropdown layer, followed by 3 fully connected layers and a final output layer. 
Since this is a regression problem, the output layer is a single continuous value that provides the predicted steering angle.

**NVIDIA Model Architecture**
![alt text][nvidia_model]


**Final Model Architecture**

For this project I have made some modifications to the original NVIDIA model. The final model architecture is described here:

First I used two layers for preprocessing. The lambda layer normalizes and mean-centers the image data between +/- 0.5. 
The following cropping layer removes the sky and the car front from the image, which are not needed for learning.

Each of the 5 convolutional layers has a 1x1 stride, and a 2x2 max pooling operation to reduce spatial resolution. 
The first 3 convolutional layers use a 5x5 filter while the final 2 use a 3x3 filter as the input dimensionality is reduced.

After the convolutional layers, a 50% dropout layer was added to prevent overfitting.

For non-linearity, [ELU activations](https://arxiv.org/pdf/1511.07289v1.pdf) are used for each convolutional and each fully connected layer.

The original NVIDIA model uses an input size of 66x200x3. I tried that, but with that I did not manage to drive the car a full round in the simulator (currently I do not know exactly why!). 
So I use the original image size of 160x320x3 as input size for my network. The cropping layer will reduce the size to 80x320x3 before the image is fed to the convolutional layers. 

Here the final model architecture:

![alt text][model_architecture]

The implementation can be found in [model.py](model.py) lines 140-176.

#### Attempts to reduce overfitting in the model
The model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py](model.py) line 39). 
And the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I used image normalization, image cropping and a dropout layer for regularization ([model.py](model.py) lines 140-176).
Also I used L2-regularisation on each convolutional and each fully connected layer to combat overfitting ([model.py](model.py) lines 157-173).

Further I used an EarlyStopping-callback to stop the training process automatically if the value-loss has not improved over some epochs. 
And a ModelCheckpoint-callback to save only the best trained model ([model.py](model.py) lines 179-188). 

#### Model parameter tuning
Here are the parameters for tuning the model and the training process. 
The tuning parameters can be adjusted in the [config.py](config.py) file.

* `NUM_EPOCHS` = 20 (Max number of epochs, since early stopping is used)
* `BATCH_SIZE` = 128
* `LERNING_RATE` = 1.0e-4
* `L2_WEIGHT`= 0.001
* `STEERING_CORRECTION` = 0.25
* `STEERING_THRESHOLD` = 0.15
* `NUM_DATA_BINS`= 21

I used an adam optimizer with an learning rate of 1.0e-4 ([model.py](model.py) line 211). 

Since this is a regression problem I used the MSE (Mean Squared Error) loss function ([model.py](model.py) line 211).

The `STEERING_CORRECTION` parameter is used to adjust the steering angles for the left (+) or right (-) camera images.

The `STEERING_THRESHOLD` parameter is used to split the driving samples where the car is driving straight ahead or cornering. This is needed for the alignment of the data distribution, explained later.

#### Appropriate training data
I used the [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity to train the model, but this data set contains some pitfalls that make it difficult to train a model that lets the car drive a complete lap in the simulator without a breakdown. 

A big problem is the extremely uneven number of records in which the car drives straight ahead (steering angle between -`STEERING_THRESHOLD` and +`STEERING_THRESHOLD`) versus the data sets where the car is cornering.

The following diagram shows the distribution of the data by the steering angle:

![alt text][distribution_1]

Another problem is that for some difficult situations, e.g. the bridge with another road surface, few sharp curves or the curve without roadside, only very few records are available compared to the normal cases. Unfortunately, it is hardly possible to identify these records.

To fix the extremely unequal number of data sets with straight-ahead driving and cornering, one could simply delete a randomly selected part of the data records with straight-ahead driving. 
However, there is the risk that the rare special cases will be deleted, which would mean that the car leaves the road in such a special case. So that's not the solution!

To avoid this problem, I group the data sets based on the steering angle in `NUM_DATA_BINS` areas between -1 and +1 using a histogram. 

After that, I'll find the group with the most records (that's where the car is driving straight ahead) and fill up each of the other groups with records randomly selected from the same group, up to the number of items of largest group * 0,75. Thus, the distribution of steering angles in the data set is compensated without losing the rare records. However, the disadvantage is that the number of data to be trained has increased massively.

The following diagram shows the distribution of the data by the steering angle, after the compensation has been made:

![alt text][distribution_2]

The data compensation is done in [data.py](data.py) lines 52-72.

#### Loading Data

In the first step, I load the data from the `driving_log.csv` file. 
Hereby I decide on the basis of the steering angle which of the 3 camera images (center, left, right) I use. 
For this I use the following rules:

* If steering angle between -`STEERING_THRESHOLD` and +`STEERING_THRESHOLD` (car drives straight ahead), I use only the center image.

* If steering angle is greater than +`STEERING_THRESHOLD` then I use the center and the left image.
 The steering angle for the left image is corrected by adding `STEERING_CORRECTION` to it. 

* If steering angle is less than -`STEERING_THRESHOLD` then I use the center image and the right image.
 The steering angle for the right image is corrected by subtracting `STEERING_CORRECTION` from it. 

The `STEERING THRESHOLD` was determined by evaluating the steering distribution histogram.

The `STEERING_CORRECTION` value was determined by a lot of trial and error.

This step is done in [data.py](data.py) lines 9-49.

After loading the data is shuffled and split into a train set and a validation set.
I use only 20% of the records for validation to avoid losing too many special training records. 

This step is done in [model.py](model.py) line 39.

#### Image Preprocessing

Since the images of the data set are stored in BGR format, but the simulator works with the RGB format, I convert the images into RGB format directly after loading [model.py](model.py) lines 48-50.

In the preprocessing step the images are normalized and mean-centered between -/+0.5.
In addition, image parts that do not contribute to driving behavior, such as the sky and the car front, are cut off.
This reduces the image size from 160x320x3 to 80x320x3. 
Both preprocessing steps are done inside the network architecture using a lambda- and a cropping-layer [data.py](data.py) lines 146-149.

|Original Image|Cropped Image (ROI)|
|-------------|-------------|
|![][sample_image]|![][cropped_image]|


#### Image Augmentation
Random Brightness
![alt text][random_brightness]

Random horizontal and vertical shift
![alt text][random_shift]

Flip horizontal
![alt text][horizontal_flip]
    
Using generator to generate randomly augmented images for each batch…
![alt text][train_image_batch]




#### Output Video

Result videos:

Track 1

|Simulator View|Camera View|
|-------------|-------------|
|[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/EpAXuGkuFr8/0.jpg)](https://youtu.be/EpAXuGkuFr8)|[![](examples/track1.jpg)](output/version-1/video.mp4)|


Challenge Track 2

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/yizCGQiDgPs/0.jpg)](https://youtu.be/yizCGQiDgPs)

---






#### 3. Creation of the Training Set & Training Process

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
