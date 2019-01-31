# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model-architecture.png "Model Visualization"
[image2]: ./examples/nVidia_model.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* data.py containing the script to load and distribute the data
* config.py defines global parameters for the project
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The `data.py` file contains the code to load the simulator data from the driving_log.csv file and to align the distribution of the data.

The `config.py` is used to define some project wide parameters, such as batch size, number of epochs, ect.

### Model Architecture and Training Strategy

#### Model Architectur

My model architecture is bases on the NVIDIA model ([End to End Learning for Self-Driving Cars Paper](https://arxiv.org/pdf/1604.07316v1.pdf)), 
because it has a fairly simple architecture and because it demonstrably provides good results for use with self-driving cars. 
It is a feed-foward layered architecture in which the output of a layer is fed into the overlying layer. 
The network consists of a normalization layer, 5 convolutional layers, a dropdown layer, followed by 3 fully connected layers and a final output layer. 
Since this is a regression problem, the output layer is a single continuous value that provides the predicted steering angle.

**NVIDIA Model Architecture**
![alt text][image2]


**Final Model Architecture**

For this project I have made some modifications to the original NVIDIA model. The final model architecture is described here:

First I used two layers for preprocessing. The lambda layer normalizes and mean-centers the image data between +/- 0.5. 
The following cropping layer removes the sky and the car front from the image, which are not needed for learning.

Each of the 5 convolutional layers has a 1x1 stride, and a 2x2 max pooling operation to reduce spatial resolution. 
The first 3 convolutional layers use a 5x5 filter while the final 2 use a 3x3 filter as the input dimensionality is reduced.

A batch normalization layer was added to each convolutional layer to accelerate learning and to reduce overfitting, see Sergey Ioffe: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

After the convolutional layers, a 50% dropout layer was added to prevent overfitting.

For non-linearity, [ELU activations](https://arxiv.org/pdf/1511.07289v1.pdf) are used for each convolutional and each fully connected layer.

The original Nvidia model uses an input size of 66x200x3. I tried that, but with that I did not manage to drive the car a full round in the simulator (currently I do not know exactly why!). 
So I use the original image size of 160x320x3 as input size for my network. The cropping layer will reduce the size to 90x320x3 before the image is fed to the convolutional layers. 
Here is a visualization of the final model architecture:


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach







The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
