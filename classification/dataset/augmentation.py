import os

import cv2
import imageio
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import re


"""
Define a ImaAug, image augmentator object, this object will perform all
declared image augmentation transformations
As defined, there will be performed over every image 4 randomly chosen and ordered
transformations
More details on each transformations, refer to comment blocks
"""

# define augmentator instance
aug = iaa.SomeOf(4, [
    # Add gaussian noise to an image, sampled once per pixel from a normal distribution N(0, s), 
    # where s is sampled per image and varies between 3% and 5% of a 255 RGB space
    iaa.AdditiveGaussianNoise(scale=(0.03 * 255, 0.05 * 255)),
    # Add random values between -25 and 25 to images, with each value being sampled once per image and then being the same for all pixel
    iaa.Add((-25, 25)),
    # change their color
    iaa.AddToHueAndSaturation((-60, 60)), 
    # Scale images to a value of 50 to 150% of their original size:
    iaa.Affine(scale=(0.5, 1.5)),
    # Rotate images by -5 to 5 degrees
    iaa.Affine(rotate=(-5, 5)),
    #  Translate images by -20 to +20% on x- and y-axis independently
    iaa.Affine(translate_percent={"x": (-0.2, 0.5), "y": (-0.2,   0.2)}),
    iaa.Crop(percent=(0, 0.2)),
    # crop and pad images
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  
    # replace one squared area within the image by a constant intensity value picked at random
    iaa.Cutout(fill_mode="constant", cval=(0, 255),
               fill_per_channel=0.5),  
    # Augmenter that sets a certain fraction of pixels in images to zero.
    iaa.Dropout(p=(0, 0.2)),
    # Transform images by moving pixels locally around using displacement fields.
    # iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
    # Blur each image with a gaussian kernel with a sigma of 3.0
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    # Adjust image contrast by scaling pixels to 255*gain*log_2(1+v/255), farily similar to Multiply
    # iaa.LogContrast(gain=(0.6, 1.4)),
    # Apply motion blur with a kernel size of 15x15 pixels to images
    iaa.MotionBlur(k=15),
    # Multiply all pixels in an image with a specific value, thereby making the image darker or brighter
    iaa.Multiply((0.5, 1.5)),
    # Multiply the saturation channel of images using random values between 0.5 and 1.5
    iaa.MultiplySaturation((0.5, 1.5)),
], random_order=True)


def augment_data(images, labels, num_iter=5):
    """
    Augments input training data
    Performs num_iter augmentations to every image
    
    Arguments:
    images: set of input images
    labels: corresponding image labes
    num_iter: integer, number of augmentation iterations performed over evert image
    defaults to 5 iterations
    
    Returns
    aug_images: set of augmented images, number of samples will end up being 
    n (original samples) * num_inter = n * num_iter
    aug_labels: set of corresponding augmented image labels
    """
    
    # we will augment an image num_iter times
    aug_images = []
    aug_labels = []
    
    for (img, label) in zip(images, labels):

        for i in range(num_iter):
            aug_img = aug(image=img)
            aug_images.append(aug_img)
            aug_labels.append(label)
        
    return aug_images, aug_labels
        