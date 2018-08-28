#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================
'''
This file stores the file names in text file
'''
import glob
import os

folders = glob.glob("*")
writeText = open('test.txt', 'w')
for folder in folders:
    files = glob.glob(folder + '/*_flair.nii.gz')
    if len(files) <= 0:
        continue
    writeText.write(os.sep + files[0] + ', ' + '\n')
writeText.close()