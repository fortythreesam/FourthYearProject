import numpy
import math
from skimage.measure import compare_psnr, compare_ssim
"""
Returns the Mean Squared Error of two images
"""
def mean_squared_error(im1, im2):

    return ((im1 - im2)**2).sum()/(im1.shape[0]*im1.shape[1])

"""
Returns the Root Mean Squared Error of two images
"""
def root_mean_squared_error(im1, im2):
    return math.sqrt(mean_squared_error(im1, im2))

"""
Returns the Peak Signal to Noise Ratio of two images
"""
def peak_signal_noise_ratio(im1, im2):
    return compare_psnr(im2, im1)

"""
returns the Image quality index of two images
"""
def image_quality_index(im1, im2):
    im1_hat = a_hat(im1)
    im2_hat = a_hat(im2)
    return (4*omega(im1,im2)*im1_hat*im2_hat)/((omega(im1,im1)+omega(im2,im2))*((im1_hat**2) + (im2_hat**2)))

def a_hat(a):
    N = a.shape[0]*a.shape[1]
    return numpy.sum(a)/N

def omega(x1, x2):
    N = x1.shape[0]*x1.shape[1]
    x1_hat = a_hat(x1)
    x2_hat = a_hat(x2)
    return numpy.sum(((x1-x1_hat)*(x2-x2_hat)))/(N-1)

"""
returns the Structural similarity index (SSIM)  of two images
"""
def structural_similarity_indix(im1, im2):
    return compare_ssim(im1, im2, multichannel=True)