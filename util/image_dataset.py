import sys
import matplotlib as plt
import cv2 as cv
import random
from util.weak_texture_patches import noise_level, weak_texture_mask
from skimage.util import random_noise
class ImageDataset:
    
    def __init__(self, size, sigma=0.0002):
        self.NUM_IMAGES = 18
        self.dir_location = "images/"
        self.base_images = []
        self.noisy_images = []
        self.weak_texture_masks = []
        self.noise_levels = []
        self.sigma = sigma
        self.load_files(size)
        
        
    def load_files(self, size):
        self.base_images = []
        numbers_used = []
        while len(self.base_images) < size:
            i = random.randint(1,self.NUM_IMAGES)
            if i in numbers_used:
                continue
            image_name = str(i) + ".jpg"
            new_image = cv.imread(self.dir_location+image_name)
            new_image = new_image[:,:,::-1]
            self.base_images += [new_image]
            noisy_image = random_noise(new_image, mode="gaussian", var=self.sigma)
            
            self.noisy_images += [noisy_image]
            try:
                nlevel, th, num = noise_level(noisy_image, conf=0.97)
                self.noise_levels += [nlevel]
                new_image_mask = weak_texture_mask(noisy_image,th)
                self.weak_texture_masks += [new_image_mask]
            except Exception as e:
                print(e)
                plt.imshow(new_image)
                plt.show()
   