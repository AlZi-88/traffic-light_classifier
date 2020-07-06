#!/usr/bin/env python
# coding: utf-8

# # Traffic Light Classifier
# ---
# 
# In this project, you’ll use your knowledge of computer vision techniques to build a classifier for images of traffic lights! You'll be given a dataset of traffic light images in which one of three lights is illuminated: red, yellow, or green.
# 
# In this notebook, you'll pre-process these images, extract features that will help us distinguish the different types of images, and use those features to classify the traffic light images into three classes: red, yellow, or green. The tasks will be broken down into a few sections:
# 
# 1. **Loading and visualizing the data**. 
#       The first step in any classification task is to be familiar with your data; you'll need to load in the images of traffic lights and visualize them!
# 
# 2. **Pre-processing**. 
#     The input images and output labels need to be standardized. This way, you can analyze all the input images using the same classification pipeline, and you know what output to expect when you eventually classify a *new* image.
#     
# 3. **Feature extraction**. 
#     Next, you'll extract some features from each image that will help distinguish and eventually classify these images.
#    
# 4. **Classification and visualizing error**. 
#     Finally, you'll write one function that uses your features to classify *any* traffic light image. This function will take in an image and output a label. You'll also be given code to determine the accuracy of your classification model.    
#     
# 5. **Evaluate your model**.
#     To pass this project, your classifier must be >90% accurate and never classify any red lights as green; it's likely that you'll need to improve the accuracy of your classifier by changing existing features or adding new features. I'd also encourage you to try to get as close to 100% accuracy as possible!
#     
# Here are some sample images from the dataset (from left to right: red, green, and yellow traffic lights):
# <img src="images/all_lights.png" width="50%" height="50%">
# 

# ---
# ### *Here's what you need to know to complete the project:*
# 
# Some template code has already been provided for you, but you'll need to implement additional code steps to successfully complete this project. Any code that is required to pass this project is marked with **'(IMPLEMENTATION)'** in the header. There are also a couple of questions about your thoughts as you work through this project, which are marked with **'(QUESTION)'** in the header. Make sure to answer all questions and to check your work against the [project rubric](https://review.udacity.com/#!/rubrics/1213/view) to make sure you complete the necessary classification steps!
# 
# Your project submission will be evaluated based on the code implementations you provide, and on two main classification criteria.
# Your complete traffic light classifier should have:
# 1. **Greater than 90% accuracy**
# 2. ***Never* classify red lights as green**
# 

# # 1. Loading and Visualizing the Traffic Light Dataset
# 
# This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:
# * 904 red traffic light images
# * 536 green traffic light images
# * 44 yellow traffic light images
# 
# *Note: All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).*

# ### Import resources
# 
# Before you get started on the project code, import the libraries and resources that you'll need.

# In[1]:


import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Training and Testing Data
# 
# All 1484 of the traffic light images are separated into training and testing datasets. 
# 
# * 80% of these images are training images, for you to use as you create a classifier.
# * 20% are test images, which will be used to test the accuracy of your classifier.
# * All images are pictures of 3-light traffic lights with one light illuminated.
# 
# ## Define the image directories
# 
# First, we set some variables to keep track of some where our images are stored:
# 
#     IMAGE_DIR_TRAINING: the directory where our training image data is stored
#     IMAGE_DIR_TEST: the directory where our test image data is stored

# In[2]:


# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# ## Load the datasets
# 
# These first few lines of code will load the training traffic light images and store all of them in a variable, `IMAGE_LIST`. This list contains the images and their associated label ("red", "yellow", "green"). 
# 
# You are encouraged to take a look at the `load_dataset` function in the helpers.py file. This will give you a good idea about how lots of image files can be read in from a directory using the [glob library](https://pymotw.com/2/glob/). The `load_dataset` function takes in the name of an image directory and returns a list of images and their associated labels. 
# 
# For example, the first image-label pair in `IMAGE_LIST` can be accessed by index: 
# ``` IMAGE_LIST[0][:]```.
# 

# In[3]:


# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# ## Visualize the Data
# 
# The first steps in analyzing any dataset are to 1. load the data and 2. look at the data. Seeing what it looks like will give you an idea of what to look for in the images, what kind of noise or inconsistencies you have to deal with, and so on. This will help you understand the image dataset, and **understanding a dataset is part of making predictions about the data**.

# ---
# ### Visualize the input images
# 
# Visualize and explore the image data! Write code to display an image in `IMAGE_LIST`:
# * Display the image
# * Print out the shape of the image 
# * Print out its corresponding label
# 
# See if you can display at least one of each type of traffic light image – red, green, and yellow — and look at their similarities and differences.

# In[4]:


#image_number = random.randint(0, len(IMAGE_LIST)-1)
image_number = 733   #directly select yellow traffic light
selected_image = IMAGE_LIST[image_number][0]
image_label = IMAGE_LIST[image_number][1]
plt.imshow(selected_image)
print("Image size: ", selected_image.shape)
print("Image label: ", image_label)


# # 2. Pre-process the Data
# 
# After loading in each image, you have to standardize the input and output!
# 
# ### Input
# 
# This means that every input image should be in the same format, of the same size, and so on. We'll be creating features by performing the same analysis on every picture, and for a classification task like this, it's important that **similar images create similar features**! 
# 
# ### Output
# 
# We also need the output to be a label that is easy to read and easy to compare with other labels. It is good practice to convert categorical data like "red" and "green" to numerical data.
# 
# A very common classification output is a 1D list that is the length of the number of classes - three in the case of red, yellow, and green lights - with the values 0 or 1 indicating which class a certain image is. For example, since we have three classes (red, yellow, and green), we can make a list with the order: [red value, yellow value, green value]. In general, order does not matter, we choose the order [red value, yellow value, green value] in this case to reflect the position of each light in descending vertical order.
# 
# A red light should have the  label: [1, 0, 0]. Yellow should be: [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called **one-hot encoded labels**.
# 
# *(Note: one-hot encoding will be especially important when you work with [machine learning algorithms](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)).*
# 
# <img src="images/processing_steps.png" width="80%" height="80%">
# 

# ---
# <a id='task2'></a>
# ### (IMPLEMENTATION): Standardize the input images
# 
# * Resize each image to the desired input size: 32x32px.
# * (Optional) You may choose to crop, shift, or rotate the images in this step as well.
# 
# It's very common to have square input sizes that can be rotated (and remain the same size), and analyzed in smaller, square patches. It's also important to make all your images the same size so that they can be sent through the same pipeline of classification steps!

# In[5]:


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image, height = 32, width = 32):
    '''This function takes as input an image returns a resized copy of the original image in the desired size.'''    
     
    standard_im = cv2.resize(image, (width, height))
    
    return standard_im
    


# ## Standardize the output
# 
# With each loaded image, we also specify the expected output. For this, we use **one-hot encoding**.
# 
# * One-hot encode the labels. To do this, create an array of zeros representing each class of traffic light (red, yellow, green), and set the index of the expected class number to 1. 
# 
# Since we have three classes (red, yellow, and green), we have imposed an order of: [red value, yellow value, green value]. To one-hot encode, say, a yellow light, we would first initialize an array to [0, 0, 0] and change the middle value (the yellow value) to 1: [0, 1, 0].
# 

# ---
# <a id='task3'></a>
# ### (IMPLEMENTATION): Implement one-hot encoding

# In[6]:


def one_hot_encode(label):
   
    '''Creates a one-hot encoded label that works for all classes of traffic lights'''
    transfer = {"red":[1, 0, 0],
                "yellow":[0, 1, 0],
                "green":[0, 0, 1],}
    one_hot_encoded = transfer[label] 
    
    return one_hot_encoded


# ### Testing as you Code
# 
# After programming a function like this, it's a good idea to test it, and see if it produces the expected output. **In general, it's good practice to test code in small, functional pieces, after you write it**. This way, you can make sure that your code is correct as you continue to build a classifier, and you can identify any errors early on so that they don't compound.
# 
# All test code can be found in the file `test_functions.py`. You are encouraged to look through that code and add your own testing code if you find it useful!
# 
# One test function you'll find is: `test_one_hot(self, one_hot_function)` which takes in one argument, a one_hot_encode function, and tests its functionality. If your one_hot_label code does not work as expected, this test will print ot an error message that will tell you a bit about why your code failed. Once your code works, this should print out TEST PASSED.

# In[7]:


# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# ## Construct a `STANDARDIZED_LIST` of input images and output labels.
# 
# This function takes in a list of image-label pairs and outputs a **standardized** list of resized images and one-hot encoded labels.
# 
# This uses the functions you defined above to standardize the input and output, so those functions must be complete for this standardization to work!
# 

# In[8]:


def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


# ## Visualize the standardized data
# 
# Display a standardized image from STANDARDIZED_LIST and compare it with a non-standardized image from IMAGE_LIST. Note that their sizes and appearance are different!

# In[9]:


image_number = random.randint(0, len(IMAGE_LIST)-1)
#image_number = 733   #directly select yellow traffic light
selected_image = STANDARDIZED_LIST[image_number][0]
image_label = STANDARDIZED_LIST[image_number][1]
plt.imshow(selected_image)
print("Image size: ", selected_image.shape)
print("Image label: ", image_label)


# # 3. Feature Extraction
# 
# You'll be using what you now about color spaces, shape analysis, and feature construction to create features that help distinguish and classify the three types of traffic light images.
# 
# You'll be tasked with creating **one feature** at a minimum (with the option to create more). The required feature is **a brightness feature using HSV color space**:
# 
# 1. A brightness feature.
#     - Using HSV color space, create a feature that helps you identify the 3 different classes of traffic light.
#     - You'll be asked some questions about what methods you tried to locate this traffic light, so, as you progress through this notebook, always be thinking about your approach: what works and what doesn't?
# 
# 2. (Optional): Create more features! 
# 
# Any more features that you create are up to you and should improve the accuracy of your traffic light classification algorithm! One thing to note is that, to pass this project you must **never classify a red light as a green light** because this creates a serious safety risk for a self-driving car. To avoid this misclassification, you might consider adding another feature that specifically distinguishes between red and green lights.
# 
# These features will be combined near the end of his notebook to form a complete classification algorithm.

# ## Creating a brightness feature 
# 
# There are a number of ways to create a brightness feature that will help you characterize images of traffic lights, and it will be up to you to decide on the best procedure to complete this step. You should visualize and test your code as you go.
# 
# Pictured below is a sample pipeline for creating a brightness feature (from left to right: standardized image, HSV color-masked image, cropped image, brightness feature):
# 
# <img src="images/feature_ext_steps.png" width="70%" height="70%">
# 

# ## RGB to HSV conversion
# 
# Below, a test image is converted from RGB to HSV colorspace and each component is displayed in an image.

# In[10]:


# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 0
#image_num = random.randint(0, len(IMAGE_LIST)-1)

#image_num = random.randint(0, len(MISCLASSIFIED)-1)

#test_im = MISCLASSIFIED[image_num][0]
#test_label = MISCLASSIFIED[image_num][1]

#test_im = STANDARDIZED_TEST_LIST[image_num][0]
#test_label = STANDARDIZED_TEST_LIST[image_num][1]

test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')


# In[11]:


def print_hsv_overview(hsv):
    '''This function uses a hsv image to print out an overview of all color channels as well as the original imiage as well.'''
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # Plot the original image and the three channels
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
    ax1.set_title('Standardized image')
    ax1.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')


# ---
# <a id='task7'></a>
# ### (IMPLEMENTATION): Create a brightness feature that uses HSV color space
# 
# Write a function that takes in an RGB image and returns a 1D feature vector and/or single value that will help classify an image of a traffic light. The only requirement is that this function should apply an HSV colorspace transformation, the rest is up to you. 
# 
# From this feature, you should be able to estimate an image's label and classify it as either a red, green, or yellow traffic light. You may also define helper functions if they simplify your code.

# In[12]:


def get_brightness_mask(hsv, test_mode = 0):
    '''This function returns a mask for detection of a brightness feature for traffic light detection.
    It uses the average brigntness and saturation value of an image to determine the limits which should be masked 
    to lockate the illuminated light.
    For testing purposes a "test_mode value could be activted. By that all masks are returned. 
    By default only the final mask is returned.'''
    #maximum brightness in the image
    b_max = np.max(hsv[:,:,2]) 
    #average brightness to get an idea of the "brightness noise" of an image
    b_min = np.sum(hsv[:,:,2])/(hsv.shape[0]*hsv.shape[1])+10 

    lower_brightness = np.array([0, 0, b_min], dtype=b_max.dtype) 
    upper_brightness = np.array([180, 255, b_max], dtype=b_max.dtype)
    
    mask1 = cv2.inRange(hsv, lower_brightness, upper_brightness)
    
    #maximum saturation in the image
    s_max = np.max(hsv[:,:,1])
    #average brightness to get an idea of the "saturation noise" of an image
    s_min = np.sum(hsv[:,:,1])/(hsv.shape[0]*hsv.shape[1])
    #some images have a very slow vaariation in brightness (fog), 
    #here the saturation value could be also a good indication of wich light is illuminated
    lower_saturation = np.array([0, s_min, 0], dtype=s_max.dtype) 
    upper_saturation = np.array([180, s_max, 255], dtype=s_max.dtype)
    
    mask2 = cv2.inRange(hsv, lower_saturation, upper_saturation)
    
    #combine the two masks to really find the traffic light and get rid of backgrond light
    #focus on brightest and saturated locations
    mask = cv2.bitwise_and(mask1, mask2)

    if test_mode == 0:
        return mask
    else:
        return mask1, mask2, mask


# In[13]:


def print_brightness_mask(hsv):
    '''Testing function to print out the brightnes mask.'''
    print("S: Min = ", np.min(hsv[:,:,1]), "Max = ",np.max(hsv[:,:,1]), "MV = ",np.sum(hsv[:,:,1])/(hsv.shape[0]*hsv.shape[1]))
    print("V: Min = ", np.min(hsv[:,:,2]), "Max = ",np.max(hsv[:,:,2]), "MV = ",np.sum(hsv[:,:,2])/(hsv.shape[0]*hsv.shape[1]))
    mask1, mask2, mask = get_brightness_mask(hsv, test_mode = 1)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.set_title('Brightness')
    ax1.imshow(mask1, cmap='gray')
    ax2.set_title('Saturation')
    ax2.imshow(mask2, cmap='gray')
    ax3.set_title('Combination')
    ax3.imshow(mask, cmap='gray')


# In[14]:


######Not needed yet, could be used in future for further precision

def get_color_mask(hsv, test_mode = 0):
    '''This function returns a mask for detection of a brightness feature for traffic light detection.
    It uses the average brigntness and saturation value of an image to determine the limits which should be masked 
    to lockate the illuminated light.
    For testing purposes a "test_mode value could be activted. By that all masks are returned. 
    By default only the final mask is returned.'''
    b_max = np.max(hsv[:,:,2])
    b_min = np.sum(hsv[:,:,2])/(hsv.shape[0]*hsv.shape[1])

    lower_brightness = np.array([0, 0, b_min], dtype=b_max.dtype) 
    upper_brightness = np.array([180, 255, b_max], dtype=b_max.dtype)
    
    mask1 = cv2.inRange(hsv, lower_brightness, upper_brightness)
    
    s_max = np.max(hsv[:,:,1])
    s_min = np.sum(hsv[:,:,1])/(hsv.shape[0]*hsv.shape[1])
    
    lower_saturation = np.array([0, s_min, 0], dtype=s_max.dtype) 
    upper_saturation = np.array([180, s_max, 255], dtype=s_max.dtype)

    mask2 = cv2.inRange(hsv, lower_saturation, upper_saturation)
    
    mask = cv2.bitwise_and(mask1, mask2)

    if test_mode == 0:
        return mask
    else:
        return mask1, mask2, mask


# In[15]:


def print_color_mask(hsv):
    '''Testing function to print out the color mask.'''
    
    print("S: Min = ", np.min(hsv[:,:,1]), "Max = ",np.max(hsv[:,:,1]), "MV = ",np.sum(hsv[:,:,1])/(hsv.shape[0]*hsv.shape[1]))
    print("V: Min = ", np.min(hsv[:,:,2]), "Max = ",np.max(hsv[:,:,2]), "MV = ",np.sum(hsv[:,:,2])/(hsv.shape[0]*hsv.shape[1]))
    mask1, mask2, mask = get_color_mask(hsv, test_mode=1)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.set_title('Brightness')
    ax1.imshow(mask1, cmap='gray')
    ax2.set_title('Saturation')
    ax2.imshow(mask2, cmap='gray')
    ax3.set_title('Combination')
    ax3.imshow(mask, cmap='gray')    


# In[16]:


def crop_sides(hsv_image):
    '''This function is used to crop the sides of an image. The aim is to focus an image to it's highest saturatio value,
    which is in this case the activated traffic light. Therefore the geometric center of the saturation value is used
    to define the column with the highest saturation. This helps the classification algorithm to focus on the light itself
    and to get rid of the image background.'''
    #Get the sum of saturation value for each column
    s_col = [np.sum(hsv_image[:,i,1]) for i in range(hsv_image.shape[1])]
    S=0
    #Get the total area of the saturation curve
    A = np.sum(s_col)
    #iterate through all the columns and sum up the area under the curve 
    for i,dA in enumerate(s_col):
        S += (i+1)*dA
    #the desire image center is the geometric center of the area under the curve (limited by the size of the image)
    center = min(int(round(S/A,0)), hsv_image.shape[1])
    #crop everything exept 5 pixels around the center
    offset = 5
    #for croping top and bottom fixed values are working fine
    r_start = 3
    r_end = -3
 
    hsv_cropped = hsv_image[r_start:r_end,center-offset:center+offset,:]
    return hsv_cropped


# In[17]:


print_hsv_overview(crop_sides(hsv))


# In[18]:


print_brightness_mask(crop_sides(hsv))


# In[19]:


print_color_mask(crop_sides(hsv))


# In[20]:


def brightest_location(rgb_image):
    '''Brightness feature for traffic light detection. The Feature searches for the activated light 
    and is mapping this information the traffic light (red -> top, yellow -> middle, greed -> bottom)'''
    #convert rgb image to hsv
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    #crop the image to get rid of the backgound
    hsv_cropped = crop_sides(hsv)
    
    #get a mask with the brightest location in the image
    mask = get_brightness_mask(hsv_cropped)
    #new image size
    height = hsv_cropped.shape[1]
    width = hsv_cropped.shape[0]
    #Sum up all rows of the mask to get the vertical location of the traffic light which is illuminated
    mask_sum = np.sum(mask[:,:], axis=1)      
    location = [0, 0, 0]
    area = height*width / 3
    #ceck for all three possible lights if the brightest location mathches with the expected position in the image
    #red -> top
    #yellow -> middle
    #greed -> bottom
    for i in range(3):
        avg = 0
        for j in range(len(mask_sum)//3):
            avg += mask_sum[i*len(mask_sum)//3 + j]
        location[i]=avg/area
    total = np.sum(location)
    if total == 0:
        total =1
    #calculation of an probability for each traffic light which is equal to illuminated area ofthe three positions 
    #(top, center, bottom) divided by the total illumination value
    p_r = location[0] / total
    p_y = location[1] / total
    p_g = location[2] / total
    return [p_r, p_y, p_g]


# In[21]:


brightest_location(test_im)


# In[22]:


def traffic_light_color(rgb_image):
    '''The color feature searches for the brightest lotion which should be the traffic light. The original image is masked 
    and the algorithm evaluetes the color of illuminated light.'''
    #conversion rgb -> hsv
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    #image cropping to get rid of the background
    hsv_cropped = crop_sides(hsv)
    #get the mask for the activated light, special color mask could be used as well instead
    mask = get_brightness_mask(hsv_cropped)
    #mask the original image to focus only on the color of the traffic light
    masked_image = np.copy(cv2.cvtColor(hsv_cropped, cv2.COLOR_HSV2RGB))

    masked_image[mask == 0] = [0, 0, 0]
    #check each color channel to define which color is most present
    r_sum = np.sum(masked_image[:,:,0])
    g_sum = np.sum(masked_image[:,:,1])
    b_sum = np.sum(masked_image[:,:,2])
    total = r_sum + g_sum + b_sum
    p_r = r_sum / total
    p_g = g_sum / total
    p_b = b_sum / total
    #currently yellow trafffic light is not evaluated since it is a mix between red and green 
    #which makes it difficult to distinguish. Absence of blue might be used in future to detect yellow
    p_y = 0
    
    return [p_r, p_y, p_g]


# In[23]:


traffic_light_color(test_im)


# ## (QUESTION 1): How do the features you made help you distinguish between the 3 classes of traffic light images?

# **Answer:**
# The features look on very specific characteristics of the activated traffic light and they try to eliminate all disturbing background "noise" of the image. With that the algorithm could focus on only very small and interesting areas of the image and the result of the decision whcih of the three classes is the right on get more and more precise

# # 4. Classification and Visualizing Error
# 
# Using all of your features, write a function that takes in an RGB image and, using your extracted features, outputs whether a light is red, green or yellow as a one-hot encoded label. This classification function should be able to classify any image of a traffic light!
# 
# You are encouraged to write any helper functions or visualization code that you may need, but for testing the accuracy, make sure that this `estimate_label` function returns a one-hot encoded label.

# ---
# <a id='task8'></a>
# ### (IMPLEMENTATION): Build a complete classifier 

# In[24]:


def estimate_label(rgb_image):
    
    brightest = brightest_location(rgb_image)
    color = traffic_light_color(rgb_image)
    #sum up all probabilities of the features and give the final decision
    combination =[p_b+p_c for p_b,p_c in zip(brightest,color)] 
    light = np.argmax(combination)
    predicted_label = [0, 0, 0]
    predicted_label[light] = 1
    
    return predicted_label   
    


# ## Testing the classifier
# 
# Here is where we test your classification algorithm using our test set of data that we set aside at the beginning of the notebook! This project will be complete once you've pogrammed a "good" classifier.
# 
# A "good" classifier in this case should meet the following criteria (and once it does, feel free to submit your project):
# 1. Get above 90% classification accuracy.
# 2. Never classify a red light as a green light. 
# 
# ### Test dataset
# 
# Below, we load in the test dataset, standardize it using the `standardize` function you defined above, and then **shuffle** it; this ensures that order will not play a role in testing accuracy.
# 

# In[25]:


# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# ## Determine the Accuracy
# 
# Compare the output of your classification algorithm (a.k.a. your "model") with the true labels and determine the accuracy.
# 
# This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing and *should not be changed*.

# In[30]:


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for i,image in enumerate(test_images):
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


# In[27]:


brightest_location(STANDARDIZED_TEST_LIST[34][0])


# ---
# <a id='task9'></a>
# ### Visualize the misclassified images
# 
# Visualize some of the images you classified wrong (in the `MISCLASSIFIED` list) and note any qualities that make them difficult to classify. This will help you identify any weaknesses in your classification algorithm.

# In[28]:


image_number = random.randint(0, len(MISCLASSIFIED)-1)   #directly select yellow traffic light
selected_image = MISCLASSIFIED[image_number][0]
image_label = MISCLASSIFIED[image_number][1]
true_label = MISCLASSIFIED[image_number][2]
plt.imshow(selected_image)
print("Image size: ", selected_image.shape)
print("Image predicted label: ", image_label)
print("Image true label: ", true_label)


# ---
# <a id='question2'></a>
# ## (Question 2): After visualizing these misclassifications, what weaknesses do you think your classification algorithm has? Please note at least two.

# **Answer:**
# - The algorithm generally has difficulties to detect yellow traffic lights since it is always in the middle between red and green
# - For very bright images the algorithm also has difficulties to differentiate between background light and traffic light 

# ## Test if you classify any red lights as green
# 
# **To pass this project, you must not classify any red lights as green!** Classifying red lights as green would cause a car to drive through a red traffic light, so this red-as-green error is very dangerous in the real world. 
# 
# The code below lets you test to see if you've misclassified any red lights as green in the test set. **This test assumes that `MISCLASSIFIED` is a list of tuples with the order: [misclassified_image, predicted_label, true_label].**
# 
# Note: this is not an all encompassing test, but its a good indicator that, if you pass, you are on the right track! This iterates through your list of misclassified examples and checks to see if any red traffic lights have been mistakenly labelled [0, 1, 0] (green).

# In[29]:


# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# # 5. Improve your algorithm!
# 
# **Submit your project after you have completed all implementations, answered all questions, AND when you've met the two criteria:**
# 1. Greater than 90% accuracy classification
# 2. No red lights classified as green
# 
# If you did not meet these requirements (which is common on the first attempt!), revisit your algorithm and tweak it to improve light recognition -- this could mean changing the brightness feature, performing some background subtraction, or adding another feature!
# 
# ---

# ### Going Further (Optional Challenges)
# 
# If you found this challenge easy, I suggest you go above and beyond! Here are a couple **optional** (meaning you do not need to implement these to submit and pass the project) suggestions:
# * (Optional) Aim for >95% classification accuracy.
# * (Optional) Some lights are in the shape of arrows; further classify the lights as round or arrow-shaped.
# * (Optional) Add another feature and aim for as close to 100% accuracy as you can get!

# In[ ]:




