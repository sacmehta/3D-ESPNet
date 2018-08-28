#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

import torch.utils.data
import nibabel as nib

'''
Custom dataset loader
'''

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, transform=None):
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image1 = nib.load(image_name).get_data() # flair
        image2 = nib.load(image_name.replace('flair', 't1')).get_data()
        image3 = nib.load(image_name.replace('flair', 't1ce')).get_data()
        image4 = nib.load(image_name.replace('flair', 't2')).get_data()
        label = nib.load(label_name).get_data()
        if self.transform:
            [image1, image2, image3, image4, label] = self.transform(image1, image2, image3, image4, label)
        return (image1, image2, image3, image4, label)
