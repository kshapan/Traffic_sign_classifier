# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./TrainingDataChart.png "Visualization"
[image4]: ./examples/2.jpg "Priority road"
[image5]: ./examples/bumpyroad.jpeg "bumpy road"
[image6]: ./examples/road_work.jpg "road work"
[image7]: ./examples/slippery_road.jpg "slipery road"
[image8]: ./examples/speed_limit_120.jpg "speed limit 120"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy/pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the dristibution of training data is.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)



As a part of preproceessing the dataset, I have used normalization method. However, when I observed the validation accuracy is improved when I used grayscaled images, I introduced grayscaling in the preprocessing of iamges. ANd then I normolized the dataset. After preprocessing the dataset is ready to training the model.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have tried several diffrent combination to come up with the final model.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64    				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x128    				|
| Fully connected		| Input 512 Output 256 							|
| RELU					|												|
| Fully connected		| Input 256 Output 128 							|
| RELU					|												|
| Fully connected		| Input 128 Output 43 							|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tried several combination of epoch, batchsize, learning rate and sigma for initial weight distribution and come up with below value to achieve better prediction.

Epoch = 10
Batchsize = 64
learning_rate = 0.001
sigma = 0.05

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy is around 0.95 
* test set accuracy is around 0.93

If an iterative approach was chosen:
* The first architecture was Lenet. It gave the validation accuracy around 0.90. I tired several combination of epoch, batchsize, learning rate and sigma, however it was never exceeding 0.92
* Then I tried dropout mechanism, but it didnt help much.
* Then I tried to add one more convolution layer and I could see that the accuracy is improved to around 0.94.
* Then I tried to change the convolution pramaters where I increase the depth of output for 28x28x6 to 28x28x32 and i could see accuracy improved to arounf 0.96 and the I ran the model over test data to get the test accuracy of 0.93.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The above traffic images might be difficult to classify because reshaping lead to loss in pixel resolution because of which one traffic sign can be persived as other.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road    		| Priority road     							| 
| Bumpy road    		| Road narrow on right							|
| road work				| Road work										|
| 120   	      		| 120       					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)



The top five soft max probabilities in percentage for traffice sign are give below

For Priority road sign
![alt text][image4]

| Probability (%)      	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| Priority road 								| 
| 0     				| Road work 									|
| 0 					| Speed limit (30km/h)							|
| 0 	      			| Speed limit (120km/h)					 		|
| 0 				    | Right-of-way at the next intersection      	|


For Bumpy road sign
![alt text][image5]

| Probability(%)       	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| Road narrows on the right   					| 
| 0     				| Bicycles crossing 							|
| 0 					| Road work 									|
| 0 	      			| Bumpy Road					 				|
| 0 				    | Wild animals crossing   						|


For Road work sign
![alt text][image6]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.46        			| Road work   									| 
| .21     				| Wild animals crossing 						|
| .17 					| Road narrows on the right						|
| .09 	      			| Slippery road 				 				|
| .06 				    | Double curve      							|


For Slipery road sign
![alt text][image7]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| Slippery road									| 
| 0     				| Dangerous curve to the left					|
| 0 					| Bicycles crossing								|
| 0 	      			| Double curve					 				|
| 0 				    | Road narrows on the right 					|


For speed limit 120 sign
![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100         			| Speed limit (120km/h)							| 
| 0     				| Speed limit (100km/h)							|
| 0 					| No vehicles									|
| 0 	      			| Vehicles over 3.5 metric tons prohibited		|
| 0 				    | Speed limit (80km/h)  						|
