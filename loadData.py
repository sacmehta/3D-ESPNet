#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# File Description: This file is used to check and pickle the data. This is adapted from my previous repository, ESPNet.
#============================================

import pickle
import nibabel as nib
import numpy as np

class LoadData:
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):

        if trainStg:
            global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                #line = textFile.read()
                line_arr = line.split(',')
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()

                label_img = nib.load(label_file).get_data()
                # There is no label with value 3 in BRATS dataset. We rename label 4 as 3.
                label_img[label_img == 4] = 3
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                if trainStg:
                    hist = np.histogram(label_img, self.classes)
                    global_hist += hist[0]
                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Some problem with labels. Please check.')
                    print('Label Image ID: ' + label_file)
                no_files += 1

        if trainStg:
            #compute the class imbalance information
            self.compute_class_weights(global_hist)

        return 0

    def processData(self):
        print('Processing training data')
        return_val = self.readFile('train.txt', True)
        
        print('Processing validation data')
        return_val1 = self.readFile('val.txt')

        print('Pickling data')
        if return_val ==0 and return_val1 ==0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList
            data_dict['classWeights'] = self.classWeights

            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None



