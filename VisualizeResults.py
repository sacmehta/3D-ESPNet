#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

'''
This file generates the results
'''

import torch
import numpy as np
import nibabel as nib
from utils import cropVolume
import Model as Net
import os

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


def segmentVoxel(imgLoc, model):
    '''
    Segment the image and store the results
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
    gth_dummy = np.copy(img_flair)
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


    output_numpy = np.zeros_like(gth_dummy)
    output_numpy[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en] = output_

    # rename the label 3 back to label 4
    output_numpy[output_numpy == 3] = 4

    # We need headers, so reload the flair image again
    img1 = nib.load(imgLoc)
    img1_new = nib.Nifti1Image(output_numpy, img1.affine, img1.header)
    name = imgLoc.replace('_flair', '').split('/')[-1]
    out_dir = './predictions'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    file_name = out_dir + os.sep + name
    nib.save(img1_new, file_name)

if __name__ == '__main__':
    data_dir = './data/test/' # evaluate on original data and not the processed one
    test_file = 'test.txt'
    if not os.path.isfile(data_dir + os.sep + test_file):
        print('Validation file not found')
        exit(-1)

    best_model_loc = './pretrained/espnet_3d_brats.pth'
    if not os.path.isfile(best_model_loc):
        print('Pretrained weight file does not exist. Please check')
        exit(-1)

    model = Net.ESPNet(classes=4, channels=4)
    model.load_state_dict(torch.load(best_model_loc))
    model = model.cuda()
    model.eval()

    dice_scores_wt = []
    dice_scores_cm = []
    dice_scores_et = []
    with open(data_dir + test_file) as txtFile:
        for line in txtFile:
            line_arr = line.split(',')
            img_file = ((data_dir).strip() + '/' + line_arr[0].strip()).strip()
            segmentVoxel(img_file, model)