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

The main goal of this project is to explore the effectiveness of automatically selecting a denoising filter and parameter. A secondary aim is to see if applying the filter separately to the weak and rich texture patches will have an effect on the outcome. The root mean squared error(RMSE), peak signal to noise ratio(PSNR), image quality index(IQI) and structural similarity index metric(SSIM) are used to evaluate the selected filters. The student aims to achieve better metrics then applying a single filter over the entire image. As a side goal, the student aims to learn more in the fields of image processing, linear algebra, and evolutionary algorithms.

### 1.3 Project Overview

Initially, the project takes in an image and applies a layer of additive white Gaussian noise to it. It then generates a mask covering the noisy images weak texture patches. The evolutionary algorithm creates a population of individuals which represent the filters to apply to the weak/rich texture patches. From here, the selected filter for each texture type is applied over a copy of the whole image and it uses the mask to take/remove the weak texture patch. The remaining patches in each denoised image are combined to give us the complete denoised image. It then measures the effectiveness of the filter by calculating one of the above metrics, RMSE for example. After enough generations of individuals are tested, it then tests the individual on a different image with a similar level of estimated noise. 

### 1.4 Achievements
___
$$\pagebreak$$
___
## Chapter 2

## Analysis

### 2.1 Image Processing



### 2.1.1 Image Denoising



### 2.1.2 Weak Texture Patches



### 1.2 Evolutionary Algorithms




## References

[1] Duran, Joan & Coll, Bartomeu & Sbert, Catalina. (2013). Chambolle's Projection Algorithm for Total Variation Denoising. Image Processing On Line. 3. 301-321. 10.5201/ipol.2013.61.     