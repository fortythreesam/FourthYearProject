# Noise Level Estimation Using Weak Textured Patches Of A Single Noisy Image

## Info

### Department of Mechanical and Control Engineering, Tokyo Institute Of Technology
Xinhao Liu
Masayuki Tanaka
Masatoshi Okutomi

## Abstract

Patches generated from a single noisy image. Can easily estimate the noise level from image patches using principle component analysis (PCA) if the image comprises only weak texture patches. Challenge for patch-based noise level estimation is how to select weak texture patches from a noisy image. A novel algorithm proposed to select weak textured patches from a from a single noisy image based on the gradients of the patches and their statistics. Then estimate the noise level from the selected weak texture patches using PCA. 

- Principle component analysis (PCA): a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linear uncorrelated variables called pronciple components
- Orthogonal Transformation: a linear transformation T: V -> V on a real product space V, that preserves the inner product. That is, for each pair u, v of elements of V, we have (u,v) = (Tu,Tv).

## Introduction

Many alg have been proposed for gray-level image noise level estimation. Generally they are classifiable into patch-based and filter-based approaches, or some combination. Proposed a patch based noise estimation algorithm using PCA and the novel texture stength metric to select weak textured patches. Investigate the relation between the proposed metric and the noise level $\sigma \small n$. Then propose an iterative framework to select weak textured patches from the noisy image with differant noise levels.

## Proposed algorithm

### Noise level estimation based on PCA

After decomposing the image into overlaping patches, the model can be written as:

$$ {y \small i} = {z \small i} + { n \small i}$$

Where $z \small i$ is the original image patch with the $i^{th}$ pixel at its center written in a vectorized format and ${y \small i}$ is the observed vectorized patch corrupted by i.i.d zero-mean Gaussian noise vector ${n \small i}$. The goal of noise level estimation is to calculate the unknown standard deviation ${\sigma \small i}$ given only the observed noisy image.


