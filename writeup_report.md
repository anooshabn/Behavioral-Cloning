# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
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
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video recording of vehicle running autonomously for two laps around track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run1
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers of 5x5 (3 layers), 3x3 (2 layers) filter sizes and four fully connected layers. (model.py lines 100-110) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 100-104). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 94-95). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### 4. Appropriate training data

I faced lot of issues with online simulator, also I kept losing data due to my bad internet connectivity. Though I downloaded simulator, collected data on my local machine, faced issued while uploading to workspace. So I ended using data provided by Udacity, also flipped the images to generate more data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure :
* the car should stay in the center of the road as much as possible
* if the car veers off to the side, it should recover back to center

My first step was to use a convolution neural network model similar to the one used in classroom which was published by autonomous vehicle team at NVIDIA. I thought this model is appropriate because it was tested in classroom and also this was the model NVIDIA used to train a real car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

model.fit_generator(train_generator, steps_per_epoch=len(train_lines), 
                    validation_data=validation_generator, validation_steps=len(validation_lines), epochs=5, verbose = 1)
                    
Initially I used 5 epochs and used the above mentioned line of code to fit the model, but it took a lot of time and due to my bad internet, it kept diconnecting and I had to train again, but never finished. So instead I modified it into below line of code after taking the suggestion on slack channel. (set the number of minibatches per epoch to the size of the dataset divided by the size of a  minibatch.)
                   
model.fit_generator(train_generator, steps_per_epoch=int(len(train_lines)/32), 
                    validation_data=validation_generator, validation_steps=int(len(validation_lines)/32), epochs=3, verbose = 1)

Classroom video on Generators really helped me to organize my project.

As a final step I ran the simulator to see how well the car was driving around track one. The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 94-95). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

#### 3. Creation of the Training Set & Training Process

As mentioned above, because of the issues I faced while collecting my own data using simulator and upoading the data, I ended using data provided by Udacity. The dataset consists of images from 3 different angles. Below are the images captured from center, left and right side of the road.

![alt text][./examples/center_2016_12_01_13_30_48_287.jpg "Center"]
![alt text][./examples/left_2016_12_01_13_30_48_287.jpg "Left"]
![alt text][./examples/right_2016_12_01_13_30_48_287.jpg "Right"]

To augment the data sat, I also flipped images and angles.

After the collection process, I had X number of data points. I then preprocessed this data by normalizing and mean centering the data by using a Lambda layer ((model.py lines 62) as follows:

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

I finally randomly shuffled the data set (model.py lines 53-55). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 94-95). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I used an adam optimizer so that manually training the learning rate wasn't necessary.
