# Image denoising using weak texture patches and genetic algorithms

$$\pagebreak$$

## Table Of Contents

1. Introduction
   1. Motivation
      1. Image Noise
      2. Evolutionary Algorithms
      3. Weak Texture Patches
   2. Goals
   3. Project Overview
   4. Achievements
2. Analysis
   1. Image Processing
      1. Image Noise
      2. Weak Texture Patches
   2. Evolutionary Algorithms
   3. Evaluation Metrics
3. Design
4. Implementation
5. Evaluation
6. Conclusions

$$\pagebreak$$

## Abstract
___

### Image denoising using weak texture patches and genetic algorithms

#### Sam Drugan

Image denoising is an important process in the fields of computer vision and digital cameras. In this project we explore how applying denoising filters, such as Chambolle’s total vision denoising algorithm and Gaussian blur, to the weak and rich textured area of an image can improve denoising. We use a genetic algorithm to select the filter and parameters that are used for each patch. Using the gradient of each patch in the image and its statistics, we generate the weak texture patches. This gives us a mask of the weak textured areas which we can use to extract the specific pixels needed using matrix multiplication. We can then combine the weak and rich texture patches, after applying the filter, to give us the complete denoised image. The evaluation methods we use are the root mean squared error, peak signal-to-noise ratio, image quality index, and structural similarity index. We tested the model on an image that had a layer of additive white Gaussian noise added. Early results show an improvement in each the previous scores when a different filter is applied to each type of patch over using a single filter on the whole image based on the estimated noise level. This could prove effective when trying to select an optimal filter to use or when trying to optimize the denoising as far as possible.


$$\pagebreak$$

## Declaration


$$\pagebreak$$

## Acknowledgements


$$\pagebreak$$

## List of Figures
___


$$\pagebreak$$

## Chapter 1

## Introduction

### 1.1 Motivation

#### 1.1.1 Image Noise

Noise is present in an image when there is a random variance in its pixel values. Image denoising is an important step in any image processing application. Fields such as computer vision rely on having high-quality images in order to work as efficiently as possible. Rafati, Mehravar, et al.[1] haveshown how sensitive such systems can be to even low levels of noise. Digital cameras are also required to output images that closely represent the real world. It is therefore essential that as much of the process of this is as efficient as possible. The presence of noise in an image can cause problems for both perceived quality and effectiveness of algorithms using the image. 

#### 1.1.2 Evolutionary Algorithm

One problem with using denoising filters is the fact that input parameters are required to be selected manually. For example, a Gaussian filter requires an estimated noise level and Chambolles total variation filter requires a weight parameter[2]. Since there are many of these filters, it can take a long time to find the optimal one to use. The idea is to use an evolutionary algorithm to automatically select the filter and parameter(s).

#### 1.1.3 Weak Texture Patches

Another problem faced in denoising images is the presence of naturally noisy areas. One example of this is a beach where the sand is naturally noisy but the sky is smooth. Many images with these areas lose much of their detail when a single filter is applied over the image. Dealing with rich/weak textured areas separately and then combining the results of the two could help address this issue.

### 1.2 Goals

The main goal of this project is to explore the effectiveness of automatically selecting a denoising filter and parameter. A secondary aim is to see if applying the filter separately to the weak and rich texture patches will have an effect on the outcome. The root mean squared error(RMSE), peak signal to noise ratio(PSNR), image quality index(IQI) and structural similarity index metric(SSIM) are used to evaluate the selected filters[3]. The student aims to achieve better metrics then applying a single filter over the entire image. As a side goal, the student aims to learn more in the fields of image processing, linear algebra, and evolutionary algorithms.

### 1.3 Project Overview

Initially, the project takes in an image and applies a layer of additive white Gaussian noise to it. It then generates a mask covering the noisy images weak texture patches. The evolutionary algorithm creates a population of individuals which represent the filters to apply to the weak/rich texture patches. From here, the selected filter for each texture type is applied over a copy of the whole image and it uses the mask to take/remove the weak texture patch. The remaining patches in each denoised image are combined to give us the complete denoised image. It then measures the effectiveness of the filter by calculating one of the above metrics, RMSE for example. After enough generations of individuals are tested, it then tests the individual on a different image with a similar level of estimated noise. 

### 1.4 Achievements
___

$$\pagebreak$$
___

## Chapter 2

## Analysis

### 2.1 Image Processing

### 2.1.1 Images in Computers

There are many ways to represent images in memory. The standard way to represent natural pictures is as a raster image. A raster image is a rectangular matrix that can vary in depth. Each position in the matrix represents that locations colour/greyscale value. In this project, images are stored in a matrix with a depth of 3 with each channel representing red, blue, and green respectively. Each position in the matrix is called a pixel. This is an additive colour model as equal amounts of each will give us white.

### 2.1.2 Image Noise/Denoising

Since storage space limits the ability to store large images due to the increasing space required by higher resolutions, a number of steps are taken to compress the data. One step is to apply colour quantisation to the image to reduce the range of values being stored. Quantisation is the process of estimating a range of values into a discrete value. This can reduce the amount of data stored but retain the same visual quality. A problem with this is it adds errors in the values to the image known as noise. Depending on the level of quantisation, the noise can be more or less noticeable.

This project will focus on additive white Gaussian noise. The main source of this type of noise is during the aquisition stage of the image due to faults in the sensor e.g. the sensors temperature is too high. A standard model for this type of noise is $X = Y + N$ where $X$ is the noisy image, $Y$ is the pure image and $N$ is the layer of additive white gaussian noise. There a various methods already available to reduce this type of noise. Listed here are a few filter types:

#### 2.1.2.1 Median Filter

Assigns the median value to the pixel of it and it's neighbours. It requires no input parameters to work. [3]

#### 2.1.2.2 Gaussian Filter

A gaussian filter blurs an image causing a reduction in noise and detail. This is achieved by convolving the image using a gaussian function. The standard deviation of the noise is required.

#### 2.1.2.3 Chambolle's Total Variation Filter 

Attempts to reduce the total variance in the image based on a given weight parameter.[2] A higher weight reduces noise further but also reduces the level of detail.

#### 2.1.2.4 Weiner Filter

Estimates the desired target image by applying a linear time-invarient filter to the signal. Similar to the Gaussian filter, it requires a noise level estimation.[3]

### 2.1.3 Weak Texture Patches

A weakly textured patch in an image is found where a cluster of pixels contains similar values to each other. Examples of this in natural images would be a wall or a clear sky. These patches are useful in noise estimation as it's easy to detect a disturbance. The main issue with this is the difficulty of detecting weak texture patches in noisy images as the noise variance causes pixel values to vary more.

Liu, Xinhao, et al.[4] propose a method to estimate noise levels of additive white Gaussian noise by analysing weak texture patches in an image. They also show a method for generating a mask of the weak texture patches in an image. The method they propose analyses statistics from the gradient covariance matrix of each patch. The process looks for what is expected in a weak texture patch after a layer of noise has affected the image. It then estimates a threshold such that a patch is weakly textured if the maximum eigenvalue of it's gradient covariance matrix falls below the threshold. This gives a matrix, with the same shape as the image, where there is a one at each position if that pixel is part of a weakly textured patch. The rest of the matrix contains zeros indicating the pixels that are part of richly textured patches. 

### 2.2 Evolutionary Algorithms

Evolutionary algorithms(EA) attempt to mimic the process of natural evolution. Sistinct components work together to emulate this process in some manner. At the most basic level, an EA will maintain a population of individuals, evaluate the effectiveness of each individual using a fitness function, and create the population of the next generation. It repeats this step for a number of generations until an optimal solution is found. After the final generation, the EA will return the overall best-performing individual as a solution. This solution is optimized to the given fitness function. It is ideal when finding solutions to problems without making assumptions about the optimal result. In particular, it is useful in this project as it allows us to search for an optimal pair of denoising filters to use on each of the texture patch types. It also allows for easy and rapid expansion of the filters/parameter combinations being used. Below is a more detailed summary of the elements in an EA:

#### 2.2.1 Individual & Population

An individual in an EA is usually represented as a bit string. The position of a sequence of bits in the string is used to represent what action to take depending on the situation. A population is the set of individuals used for any given generation.

#### 2.2.2 Evaluation & Fitness

The evaluation step takes each individual in the population and returns a fitness value. This step will vary the most between different uses. In the case of this project, it will measure the effectiveness of the denoising filter.

#### 2.2.3 Selection, Crossover and Mutating

There are a number of different techniques for the selection of individuals that are used to create a population for the next generation. One example of a selection function that this project uses is tournament selection. Tournament selection randomly takes a small group of individuals and returns the best performing amongst them. The crossover step pairs up the selected individuals and creates two new individuals. It creates the new individuals by applying a point crossover on the bits strings and returning the two possible results. Finally, there is a chance, decided by the user, for an individual in the next population to be mutated in some way. One common way to do this is to flip one bit in the bit string.

### 2.3 Evaluation Metrics

As shown above, we have four methods of evaluating the effectiveness of a denoising filter. The main reason for this is that the EA optimizes the result based on the fitness function. One of the above metrics will be used as a the fitness function and we use the others to analyse it's effectiveness in other areas.

#### 2.3.1 Root Mean Squared Error (RMSE)

RMSE gives us the root of the average difference between pixel values in the original image and the denoised image.

$$ MSE = {1\over{MN}}{\sum^M_{i=1}}{\sum^N_{j=1}}(x(i,j) - y(i,j) )^2 $$

$$ RMSE = \sqrt{MSE}$$

#### 2.3.2 Peak Signal To Noise Ratio (PSNR)

$$PSNR = 10\log_{10}{{(2^n - 1)^2}\over{\sqrt{MSE}}}$$

#### 2.3.3 Image Quality Index (IQI)

$$IQI = { 4*\sigma_{xy}*\bar{x}*\bar{y}\over{(\sigma^2_x + \sigma^2_y)*((\bar{x})^2 + (\bar{y})^2)} }$$

#### 2.3.4 Structural Similarity Index Metric SSIM

$$SSIM = {(2*\bar x * \bar y + C_1)(2*\sigma_{xy} + C_2) \over (\sigma^2_x + \sigma^2_y + C_2)*((\bar{x})^2 + (\bar{y})^2 + C_1)}$$

$$\bar x = {1\over N}\sum^N_{i=1} x_i$$

$$\bar y = {1\over N}\sum^N_{i=1} y_i$$

$$\sigma^2_x = {1 \over N - 1} \sum^N_{i=1}(x_i-\bar x)^2$$

$$\sigma^2_y = {1 \over N - 1} \sum^N_{i=1}(y_i-\bar y)^2$$

$$\sigma_{xy} = {1 \over N - 1} \sum^N_{i=1}(x_i-\bar x)(y_i-\bar y)$$


## References

[1]  Hossein Hosseini, Baicen Xiao, and Radha Poovendran.  Google’scloud  vision  api  is  not  robust  to  noise.   InMachine  Learning  andApplications  (ICMLA),  2017  16th  IEEE  International  Conference  on,pages 101–105. IEEE, 2017

[2] Duran, Joan & Coll, Bartomeu & Sbert, Catalina. (2013). Chambolle's Projection Algorithm for Total Variation Denoising. Image Processing On Line. 3. 301-321. 10.5201/ipol.2013.61.     

[3] Rafati, Mehravar, et al. “Fuzzy Genetic-Based Noise Removal Filter for Digital Panoramic X-Ray Images.” Biocybernetics and Biomedical Engineering, vol. 38, no. 4, 2018, pp. 941–965., doi:10.1016/j.bbe.2018.08.005.

[4] Liu, Xinhao, et al. “Single-Image Noise Level Estimation for Blind Denoising.” IEEE Transactions on Image Processing, vol. 22, no. 12, 2013, pp. 5226–5237., doi:10.1109/tip.2013.2283400.