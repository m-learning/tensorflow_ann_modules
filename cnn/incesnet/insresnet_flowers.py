'''
Created on Sep 23, 2016

Retraining of inception-resnet for flowers data set

@author: Levan tsinadze
'''
from cnn.flowers import cnn_files

#Prepares flowers for inception
def defile_parameters():
  file_mngr = cnn_files()
  file_mngr.get_or_init_training_set()
  
  