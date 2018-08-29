#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

'''
This file computes the dice score
'''

import torch
import numpy as np
import nibabel as nib
from utils import cropVolume
import Model as Net
import os
import copy

def norm(im):
    '''
    Normalize the image using min-max normalization
    :param im: Input image
    :return: Min-max normalized image
    '''
    im = im.astype(np.float32)

    # min-max normalization [0,1]
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v)
    return im

def cropVolumes(img1, img2, img3, img4):
    '''
    This function crop the 4 volumes that BRATS dataset provides
    :param img1: Volume 1
    :param img2: Volume 2
    :param img3: Volume 3
    :param img4: Volume 4
    :return: maximum dimensions across three dimensions
    '''
    ch_s1, ch_e1, wi_s1, wi_e1, hi_s1, hi_e1 = cropVolume(img1, True)
    ch_s2, ch_e2, wi_s2, wi_e2, hi_s2, hi_e2 = cropVolume(img2, True)
    ch_s3, ch_e3, wi_s3, wi_e3, hi_s3, hi_e3 = cropVolume(img3, True)
    ch_s4, ch_e4, wi_s4, wi_e4, hi_s4, hi_e4 = cropVolume(img4, True)

    # find the maximum dimensions
    ch_st = min(ch_s1, ch_s2, ch_s3, ch_s4)
    ch_en = max(ch_e1, ch_e2, ch_e3, ch_e4)
    wi_st = min(wi_s1, wi_s2, wi_s3, wi_s4)
    wi_en = max(wi_e1, wi_e2, wi_e3, wi_e4)
    hi_st = min(hi_s1, hi_s2, hi_s3, hi_s4)
    hi_en = max(hi_e1, hi_e2, hi_e3, hi_e4)

    return wi_st, wi_en, hi_st, hi_en, ch_st, ch_en

def diceFunction(im1, im2):
    '''
    Compute the dice score between two input images or volumes. Note that we use a smoothing factor of 1.
    :param im1: Image 1
    :param im2: Image 2
    :return: Dice score
    '''
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() + 1) / (im1.sum() + im2.sum() + 1)

def get_whole_tumor_mask(data):
    # this function returns the Whole tumor mask (i.e. all labels are merged)
    return data > 0

def get_tumor_core_mask(data):
    # This function merges label 1 and 3 to get the Core Mask
    return np.logical_or(data == 1, data == 3)

def get_enhancing_tumor_mask(data):
    # This function returns the mask for enhancing tumor i.e. label 3
    # Note that in original files, there is no mask element with value 4. We have renamed label 4 to label 3 in
    # our transformations.
    return data == 3

def computeDiceScores(im1, im2):
    # This function computes the dice score for Whole tumor, core mask, and enhancing tumor.
    im1_wt = get_whole_tumor_mask(im1)
    im2_wt = get_whole_tumor_mask(im2)

    im1_cm = get_tumor_core_mask(im1)
    im2_cm = get_tumor_core_mask(im2)

    im1_et = get_enhancing_tumor_mask(im1)
    im2_et = get_enhancing_tumor_mask(im2)

    d_wt = diceFunction(im1_wt, im2_wt)
    d_cm = diceFunction(im1_cm, im2_cm)
    d_et = diceFunction(im1_et, im2_et)

    return d_wt, d_cm, d_et


def segmentVoxelAndComputeDice(imgLoc, model):
    '''
    SEgment the image and compute the dice scores
    :param imgLoc: location of the flair image. Other modalities locations can be computed using this one.
    :param model: Segmentation Network
    :return: dice scores
    '''
    t1_loc = imgLoc.replace('flair', 't1')
    t1ce_loc = imgLoc.replace('flair', 't1ce')
    t2_loc = imgLoc.replace('flair', 't2')
    label_loc = imgLoc.replace('flair', 'seg')

    # get the 4 modalities
    img_flair = nib.load(imgLoc).get_data()
    img_t1 = nib.load(t1_loc).get_data()
    img_t1ce = nib.load(t1ce_loc).get_data()
    img_t2 = nib.load(t2_loc).get_data()

    # Crop and min-max normalize them
    wi_st, wi_en, hi_st, hi_en, ch_st, ch_en = cropVolumes(img_flair, img_t1, img_t1ce, img_t2)
    img_flair = norm(img_flair[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    img_t1 = norm(img_t1[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    img_t1ce = norm(img_t1ce[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])
    img_t2 = norm(img_t2[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en])


    # convert to Tensor
    resize = (1, img_flair.shape[0], img_flair.shape[1], img_flair.shape[2])
    img_flair = img_flair.reshape(resize)
    img_t1 = img_t1.reshape(resize)
    img_t1ce = img_t1ce.reshape(resize)
    img_t2 = img_t2.reshape(resize)

    tensor_flair = torch.from_numpy(img_flair)
    tensor_t1 = torch.from_numpy(img_t1)
    inputB = torch.from_numpy(img_t1ce)
    tensor_t2 = torch.from_numpy(img_t2)
    del img_flair, img_t1, img_t1ce, img_t2

    # concat the tensors and then feed them to the model
    tensor_concat = torch.cat([tensor_flair, tensor_t1, inputB, tensor_t2], 0)  # inputB #
    tensor_concat = torch.unsqueeze(tensor_concat, 0)
    tensor_concat = tensor_concat.cuda()

    #convert to variable.
    # If you are using PyTorch version > 0.3, then you don't need this
    tensor_concat_var = torch.autograd.Variable(tensor_concat, volatile=True)

    # Average ensembling at multiple resolutions
    # We found by experiments that ensembling at original and 144x144x144 gives best results.
    test_resolutions = [None, (144, 144, 144)]
    output = None
    for res in test_resolutions:
        if output is not None:
            # test at the scaled resolution and combine them
            output = output + model(tensor_concat_var, inp_res=res)
        else:
            # test at the original resolution
            output = model(tensor_concat_var, inp_res=res)
    output = output / len(test_resolutions)
    del tensor_concat, tensor_concat_var

    # convert the output to segmentation mask and move to CPU
    output_ = output[0].max(0)[1].data.byte().cpu().numpy()


    # load the Ground truth
    gth = nib.load(label_loc).get_data()
    gth = gth.astype(np.byte)
    # BRATS does not have label 3 and we renamed label 4 as 3 during training.
    # doing it here for consistency and computing the correct scores.
    gth[gth == 4] = 3


    output_numpy = np.zeros_like(gth)
    output_numpy[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en] = output_

    #compute the dice scores
    d_wt, d_cm, d_et = computeDiceScores(output_numpy, gth)

    return d_wt, d_cm, d_et

if __name__ == '__main__':
    # Change these variables depending upon your settings
    data_dir = './data/original_brats17/'  # evaluate on original data and not the processed one
    val_file = 'val.txt'
    best_model_loc = './pretrained/espnet_3d_brats.pth'

    if not os.path.isfile(data_dir + os.sep + val_file):
        print('Validation file not found')
        exit(-1)

    if not os.path.isfile(best_model_loc):
        print('Pretrained weight file does not exist. Please check')
        exit(-1)
    model = Net.ESPNet(classes=4, channels=4)
    # load the pretrained model
    model.load_state_dict(torch.load(best_model_loc))
    model = model.cuda()
    model.eval()

    dice_scores_wt = []
    dice_scores_cm = []
    dice_scores_et = []
    with open(data_dir + val_file) as txtFile:
        for line in txtFile:
            line_arr = line.split(',')
            img_file = ((data_dir).strip() + '/' + line_arr[0].strip()).strip()
            d_wt, d_cm, d_et = segmentVoxelAndComputeDice(img_file, model)
            dice_scores_wt.append(d_wt)
            dice_scores_cm.append(d_cm)
            dice_scores_et.append(d_et)
    print('Mean Dice Score (WT): ', np.mean(dice_scores_wt))
    print('Mean Dice Score (CM): ', np.mean(dice_scores_cm))
    print('Mean Dice Score (ET): ', np.mean(dice_scores_et))



