#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================
'''
This file generates a mapping between FLAIR file and segmentation file
'''
import glob
import os

folders = ['HGG', 'LGG']
writeText = open('train.txt', 'w')
for folder in folders:
    sub_folders = glob.glob(folder + os.sep + '*')
    for sub_folder in sub_folders:
        files = glob.glob(sub_folder + '/*_flair.nii.gz')
        if len(files) <= 0:
            continue
        writeText.write(os.sep + files[0] + ', ' + os.sep + files[0].replace('flair', 'seg') + '\n')
writeText.close()
