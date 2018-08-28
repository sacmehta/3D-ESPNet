#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

import numpy as np

def cropVolume(img, data=False):
    '''
    Helper function to remove the redundant black area from the 3D volume
    :param img: 3D volume
    :param data: Nib allows you to access 3D volume data using the get_data(). If you have already used it before
    calling this function, then it is false
    :return: returns the crop positions acrss 3 axes (channel, width and height)
    '''
    if not data:
       img = img.get_data()
    sum_array = []


    for ch in range(img.shape[2]):
        values, indexes = np.where(img[:, :, ch] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    ch_s = np.nonzero(sum_array)[0][0]
    ch_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[0]):
        values, indexes = np.where(img[width, :, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    wi_s = np.nonzero(sum_array)[0][0]
    wi_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[1]):
        values, indexes = np.where(img[:, width, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    hi_s = np.nonzero(sum_array)[0][0]
    hi_e = np.nonzero(sum_array)[0][-1]

    return ch_s, ch_e, wi_s, wi_e, hi_s, hi_e