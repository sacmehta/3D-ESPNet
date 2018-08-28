#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# File Description: This file contains the CNN models and is adapted from ESPNet and Y-Net
# ESPNET: https://arxiv.org/pdf/1803.06815.pdf
# Y-Net: https://arxiv.org/abs/1806.01313
# ==============================================================================

import numpy as np
import torch
import random
import skimage.transform as skTrans

class ScaleToFixed(object):
    """
    All images after removing redundard black pixels are of different sizes.
    Fix their size, so that we can group them in batches
    """
    def __init__(self, dimA, dimB, dimC):
        self.dimA = dimA
        self.dimB = dimB
        self.dimC = dimC

    def __call__(self, image, imageA, imageB, imageC, label):
        image = skTrans.resize(image, (self.dimA, self.dimB, self.dimC), order=1, preserve_range=True)  #
        imageA = skTrans.resize(imageA, (self.dimA, self.dimB, self.dimC), order=1, preserve_range=True)  #
        imageB = skTrans.resize(imageB, (self.dimA, self.dimB, self.dimC), order=1, preserve_range=True)  #
        imageC = skTrans.resize(imageC, (self.dimA, self.dimB, self.dimC), order=1, preserve_range=True)  #

        label = skTrans.resize(label, (self.dimA, self.dimB, self.dimC), order=0, preserve_range=True)

        return [image, imageA, imageB, imageC, label]
    
class RandomFlip(object):
    """Randomly flips (horizontally as well as vertically) the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, imageA, imageB, imageC, label):

        if random.random() < 0.5:
            flip_type = np.random.randint(0, 3) # flip across any 3D axis

            image = np.flip(image, flip_type)
            imageA = np.flip(imageA, flip_type)
            imageB = np.flip(imageB, flip_type)
            imageC = np.flip(imageC, flip_type)
            label = np.flip(label, flip_type)

        return [image,imageA, imageB, imageC, label]


class MinMaxNormalize(object):
    """Min-Max normalization
    """
    def __call__(self, image, imageA, imageB, imageC, label):
        def norm(im):
            im = im.astype(np.float32)
            min_v = np.min(im)
            max_v = np.max(im)
            im = (im - min_v)/(max_v - min_v)
            return im
        image = norm(image)
        imageA = norm(imageA)
        imageB = norm(imageB)
        imageC = norm(imageC)

        return [image,imageA, imageB, imageC, label]


class ToTensor(object):

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, imageA, imageB, imageC, label):

        #image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        imageA = imageA.astype(np.float32)
        imageB = imageB.astype(np.float32)
        imageC = imageC.astype(np.float32)
        
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        imageA = imageA.reshape((1, imageA.shape[0], imageA.shape[1], imageA.shape[2]))
        imageB = imageB.reshape((1, imageB.shape[0], imageB.shape[1], imageB.shape[2]))
        imageC = imageC.reshape((1, imageC.shape[0], imageC.shape[1], imageC.shape[2]))
        
        dims = label.shape

        label = skTrans.resize(label, (int(dims[0]/self.scale), int(dims[1]/self.scale), int(dims[2]/self.scale)),
                                order=0, preserve_range=True)
        # rename label 3 as 4
        label[label == 4] = 3

        image_tensor = torch.from_numpy(image)
        image_tensorA = torch.from_numpy(imageA)
        image_tensorB = torch.from_numpy(imageB)
        image_tensorC = torch.from_numpy(imageC)
        
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int))  # torch.from_numpy(label)

        return [image_tensor,image_tensorA, image_tensorB, image_tensorC, label_tensor]


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
