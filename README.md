
# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/udacity_car_not_car.png
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/spatial_features.png
[image3]: ./output_images/color_hist_features.png
[image4]: ./output_images/hog_features.png
[image5]: ./output_images/boxes.png
[image6]: ./output_images/multiscale_boxes.png
[image7]: ./output_images/pipeline_result.png
[image8]: ./output_images/heat_map_result.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
I will consider the rubric points and describe how I solved this project. 

---
## Data Preparation

The first part of my [project code notebook](./project_code.ipynb) deals exclusively with setting up all the data for training the classifier, which will later on be used to predict on the output of sliding windows if a car is present or not. But we're getting ahed of us. Let's start by discussing which data I used.

I planned on using two data sources:

* The data provided by udacity for this project which came with [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.
* Additional data, the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) which had to be prepared fisrt.

In the first cell of my notebook I read in the provided `vehicle` and `non-vehicle` data. I examined the data and choose a random sample to display:

`Your function returned a count of 8792  cars and 8968  non-cars
of size:  (64, 64, 3)  and data type: uint8`

![alt text][image1]

Next came the `Udacity labeled dataset`. Since it comes with a bunch of images and a csv pointing out where the bounding boxes are for each label (cars, pedestrians and trucks in my case) it had to be prepared first (see cell two of my notebook). I decided to extract the image regions corresponding with car labels only. In the future using also trucks and pedestrians the classifier will be of more use to a self-driving car because different actions are required for each of those object when they appear in the path. I used pandas to prepare the dataframe.

But wait a minute, the provided data had `vehicle` and `non-vehicle` data nicely balanced and all 64x64 pixels. I decided to build a function that randomly defines a region on the image and takes that as `non-vehicle` data for each `vehicle` image extracted. Keep in mind that a picture may have multiple cars in it. That's why it is important to check whether the random region is within any of the already assigned regions. I then extracted every `Car` labeled region and a random region. While extracting I went ahead and resized to 64x64 pixels. The extracted images (ready to go, read in with `cv2.imread` so on a 0-255 scale) were saved to two pickle files.

Cell three of my notebook is just used to display samples from the udacity labeled dataset:

![alt text][image0]

In cell 4 I'm bringing together the two datasets. My first idea was to append all images from teh udacity labeled dataset. That may have been a bit overkill. I decided it's not worth it and reduced the size. I experimented using 5000 random images picked trom the `udacity labeled dataset` and appended to our original data (which I also read in using cv2.imread). However, after experimenting for a while I decided to not add any additional data (only 1 to show how it can be done). The results were just better. A possible reason may be that the `udacity labeled dataset` has images with parked cars etc. All those images which are now ready to be used as training data for the classifier were shuffled and then saved to two pickle files.

In the next cell I'm getting rid of all those variables I didn't need anymore. Form here on out I just used the pickle data directly. 

Note: I have multiple sections in this notebook which can be run independently from the rest, that leads to a lot of duplication in code. Thinking about it now, I should have probably split this into single files. Anyway, thats why the notebook is quite long and may seem bulky.


## Feature extraction (Spatial Binning of Color, Histograms of Color, Histogram of Oriented Gradients (HOG)) and classifier training (spoiler alert, it's a SVM)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. (+ spatial binned color features and histograms of color)

The code for this step is contained in the second section of my IPython notebook (cells 6 through 11).  

I started by reading in all the prepared `vehicle` and `non-vehicle` images. I then explored 3 options to generate features. Please keep in mind that I tried different color spaces. For my final pipeline I went with LUV after having used YCrCb for some time before. This is the data information:

`Your function returned a count of 8793  cars and 8969  non-cars
of size:  (64, 64, 3)  and data type: uint8`

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
Here is an example using a grayscaled image and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

![alt text][image4]

The second option is just straight up color features. Using `imresize` to sclae the image and `ravel()` to create a feature vector. The output (for BGR colorspace and 20x20 dimensions) looks like this: 

![alt text][image2]

The third option are color histogram values. The parameter to tune here is the number of bins. I went with 32 for some time, but read an [article](https://medium.com/towards-data-science/vehicle-detection-and-distance-estimation-7acde48256e1) which favored 128. My results with trhis configuration were really good. Here is a sample histogram for color channels `BGR` (all three histograms are displayed in a single graph, 128 bins each):

![alt text][image3]

Cells 10 and 11 contain code to extract all features from our `cars` and `notcars` lists. The image features are extracted and scaled. The scaler, feature veactor and label vector are saved to pickle files.

The features were extracted using the following parameters:

* color_space = 'LUV' # Can be BGR, HSV, LUV, HLS, YUV, YCrCb
* orient = 12  # HOG orientations
* pix_per_cell = 8 # HOG pixels per cell
* cell_per_block = 1 # HOG cells per block
* hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
* spatial_size = (20, 20) # Spatial binning dimensions
* hist_bins = 128    # Number of histogram bins
* spatial_feat = True # Spatial features on or off
* hist_feat = True # Histogram features on or off
* hog_feat = True # HOG features on or off

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and colorspaces for my pipeline and read articles online. I finally decided on increasing the number of orientations to 12 (giving better differentiation options) and decreasing the ceels per block to 1. Most of this process was iterative and based on experimenting and heuristics.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is contained in cell 12 of my notebook.

I started my training process with a SVM using `GridSearchCV` with `parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 0.1, 1e-4]}` making 8 options, fitted with 3 folds per option totalling in 24 folds. At some point the rbf kernel with C=10 gave me the best results, and I almost kept using it. However, the time to predict was a too high. I then switched to LinearSVC with C= 1e-4. 

My final results on the test set (10% of total data):

`Test Accuracy of SVC =  0.9949
My SVC predicts:  [ 1.  1.  0.  0.  0.  0.  0.  0.  0.  1.]
For these 10 labels:  [ 1.  1.  0.  0.  0.  0.  0.  0.  0.  1.]
0.01906 Seconds to predict 10 labels with SVC`

The classifier was saved to pickle for later use.


## Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for a sliding window search (+ some other functions, e.g. to draw boxes) is included in cell 14 of my project_code notebook. Most of the code was adapted from the udacity course materials.

Sliding 64x64 pixels windows with an overlap of 0.5 for x and y direction, searching in the lower half of the image produces the following windows:

![alt text][image5]

To achieve better results I choose different search window sizes. I staret experimenting on my own, but ended up adapting the image sizes from this [article](https://medium.com/towards-data-science/vehicle-detection-and-distance-estimation-7acde48256e1). The search was canducted in the lower half of the image, excluding the small region on the bottom where the car's hood can be seen. I did not adapt for distance, which will most likely help to decrease processing time and may help in further reducting false positives. Smaller search image sizes would only be used for the regions which are further away, bigger search windows for regions more close up.

The result of applying all of my search windows looks like this:

![alt text][image6]

Please see project code cell 16 for more details.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on six scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example images:

![alt text][image7]

Performace was optimized through iterations of trying different features using different colorspaces and tuning the parameters, as well as search windows and scales.

---

## Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from an example image and the bounding boxes then overlaid:

![alt text][image8]

#### Additional Notes:

My notebook contains two final pipelines, which use all the single steps explored throughout the first parts of the notebook. These pipelines can be found in cell 21 and 23. The first pipeline uses the same techniques as explored before, but is slow. To speed things up, the second pipeline uses hog subsampling.

---

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As mentioned above, selecting the right training data and classifier caused me some trouble. At some point I had a pipeline which took between 50 and 86 seconds to process a single frame. As I found out later, this was mostly due to using a 'rbf' kernel classifier.

Extending the dataset is generally needed to allow for more robust classifiers, especially for city driving with parked cars, pedestrians and also trucks (which will currently not be identified).

My pipeline will fail if two cars are directly behind each other (obviously). One could assume that cars don't just disappear and use color to track cars even if they are behind each other. Smoothing the bounding boxes would alo be nice. They can be used to calculate distance and realtive speed.

Nighttime driving will pose a challenge, since colors may be very different (closer to grayscale). 

My next step for when I have more time is to use a pretrained neural net and use it for image classification. The tensorflow object-detection implementation which was recently released is an ideal candidate. Using this method will most likely also help with processing times. My current implementation (at least with my processor) would't be safe to use for a real world implementation.

