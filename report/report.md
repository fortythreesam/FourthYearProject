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
   3. Evaluation Methods
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

Noise is present in an image when there is a random variance in its pixel values. Image denoising is an important step in any image processing application. Fields such as computer vision rely on having high-quality images in order to work as efficiently as possible. Digital cameras are also required to output images that closely represent the real world. It is therefore essential that as much of the process of this is as efficient as possible. The presence of noise in an image can cause problems for both perceived quality and effectiveness of algorithms using the image. 

#### 1.1.2 Evolutionary Algorithm

One problem with using denoising filters is the fact that input parameters are required to be selected manually. For example, a Gaussian filter requires an estimated noise level and Chambolles total variation filter requires a weight parameter[1]. Since there are many of these filters, it can take a long time to find the optimal one to use. The idea is to use an evolutionary algorithm to automatically select the filter and parameter(s).

#### 1.1.3 Weak Texture Patches

Another problem faced in denoising images is the presence of naturally noisy areas. One example of this is a beach where the sand is naturally noisy but the sky is smooth. Many images with these areas lose much of their detail when a single filter is applied over the image. Dealing with rich/weak textured areas separately and then combining the results of the two could help address this issue.

### 1.2 Goals

The main goal of this project is to explore the effectiveness of automatically selecting a denoising filter and parameter. A secondary aim is to see if applying the filter separately to the weak and rich texture patches will have an effect on the outcome. The root mean squared error(RMSE), peak signal to noise ratio(PSNR), image quality index(IQI) and structural similarity index metric(SSIM) are used to evaluate the selected filters[2]. The student aims to achieve better metrics then applying a single filter over the entire image. As a side goal, the student aims to learn more in the fields of image processing, linear algebra, and evolutionary algorithms.

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

This project will focus on additive white Gaussian noise. The main source of this type of noise is during the aquisition stage of the image due to faults in the sensor e.g. the sensors temperature is too high. We can model this type of noise as $X = Y + N$ where $X$ is the noisy image, $Y$ is the pure image and $N$ is the layer of additive white gaussian noise. There a various methods already available to reduce this type of noise. Listed here are a few filter types:

#### 2.1.2.1 Median Filter

Assigns the median value to the pixel of it and it's neighbours. It requires no input parameters to work. [2]

#### 2.1.2.2 Gaussian Filter

A gaussian filter blurs an image causing a reduction in noise and detail. This is achievd by convolving the image using a gaussian function. The standard deviation of the noise is required.

#### 2.1.2.3 Chambolle's Total Variation Filter 

Attempts to reduce the total variance in the image based on a given weight parameter. [1]

#### 2.1.2.4 Weiner Filter

Estimates the desired target image by applying a linear time-invarient filter to the signal. Similar to the Gaussian filter, it requires a noise level estimation.

### 2.1.3 Weak Texture Patches



### 1.2 Evolutionary Algorithms




## References

[1] Duran, Joan & Coll, Bartomeu & Sbert, Catalina. (2013). Chambolle's Projection Algorithm for Total Variation Denoising. Image Processing On Line. 3. 301-321. 10.5201/ipol.2013.61.     

[2] Rafati, Mehravar, et al. “Fuzzy Genetic-Based Noise Removal Filter for Digital Panoramic X-Ray Images.” Biocybernetics and Biomedical Engineering, vol. 38, no. 4, 2018, pp. 941–965., doi:10.1016/j.bbe.2018.08.005.