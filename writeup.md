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

[image2]: ./examples/center_2018_06_11_22_40_23_627.jpg "Center Lane Driving"
[image3]: ./examples/center_2018_06_11_22_46_30_919.jpg "Recover Driving 1"
[image4]: ./examples/center_2018_06_11_22_46_31_132.jpg "Recover Driving 2"
[image5]: ./examples/center_2018_06_11_22_46_31_847.jpg "Recover Driving 3"
[image6]: ./examples/center1.jpg "Normal Image"
[image7]: ./examples/center2.jpg "Flipped Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 video for the car autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x32x3 RGB image   							|
| Lambda				| normalized the data							|
| Cropping				| filter out no use data						| 
| Convolution 1     	| apply a 5x5 convolution with 24 output filters|
| RELU					|												|
| Convolution 1         | apply a 5x5 convolution with 36 output filters|
| RELU                  |                                               |
| Convolution 1         | apply a 5x5 convolution with 48 output filters|
| RELU                  |                                               |
| Convolution 1         | apply a 3x3 convolution with 64 output filters|
| RELU                  |                                               |
| Convolution 1         | apply a 3x3 convolution with 64 output filters|
| RELU                  |                                               |
| Flatten    			|  												|
| Fully connected		| output 100    								|
| Fully connected		| output 50    									|
| Fully connected		| output 10    									|
| Fully connected		| output 1    									|

At beginning the data is normalized by a Lambda layer
After that I applied a Cropping layer to only focus on the data which we cared 
Then we have 5 convolution neural network with 
1. a 5x5 filter sizes and depths with 24 
2. a 5x5 filter sizes and depths with 36
3. a 5x5 filter sizes and depths with 48  
4. a 3x3 filter sizes and depths with 64 
5. a 3x3 filter sizes and depths with 64 
(model.py lines 70-74)
After every convolution layer, I applied a RELU layer to introduce nonlinearity 
After that I applied a flatten layer and four full connected layer


#### 2. Attempts to reduce overfitting in the model

During the training period I didn't very obvious overfitting result, so I didn't use dropout layer in this project.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
1. I used a center lane driving, 
2. I tried recovering from the left and right sides of the road,
3. For the place which easy to drive out, I collect more data, 
4. I slow down the speed when there is a turn

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get a easy and fast trained model with a high accuracy.  

My first step was to use a convolution neural network model similar to the Lenet, I thought this model might be appropriate because Lenet is quite powerful and easy to implement, for a first try it is a good choice.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I tried to add one more convolution layer and one dropout layer, but the result not change to much. 

Then I add a Lambda layer to get the normalized and mean centered data. This time the overfitting issue seems like been solved, however the result still not good, when I tested using the simulator the car kept turning left and right, it was not smooth. And after few second it drive out of the road

Then I follow the slids in this lesson to flip images and steering measurements, then the result become better.

After that I also add the left and right image for model training and add cropping layer in the model to focus on the data we want. The result becomes better and better.

In the end I decied to try the model which is used by Nvida for real autonomous project. It was amazing and the car runs quite smooth then before.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, and some place even cannot cover back, to improve the driving behavior in these cases, I do two steps
1. For the place which easy to drive out, I collect more data
2. I slow down the speed when there is a turn. Because I found the simulator record the image by a fix time interval, if I drive the car too fast, there will be not enought image for the turn case. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture I have state quite clearly in last part. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back, when the car drive to out of the road. These images show what a recovery looks like starting from car nearly go out of the road, give a quite big steering value to drive the car back to the center of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would give more smaples to the model and help the model become generlized. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Then I also add the left and right camera image to the data set with an offset of the steering measurements. This will provide more data to the model.

After the collection process, I had 10127x6 number of data points. I then preprocessed this data by a Lambda layer


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss rate will not change too much after epochs bigger then 5.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
