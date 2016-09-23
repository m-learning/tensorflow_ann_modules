'''
Created on Sep 23, 2016

Retraining of inception-resnet for flowers data set

@author: Levan tsinadze
'''
from cnn.flowers import cnn_files
from cnn.incesnet.config_parameters import define_training_parameters, define_eval_parameters 

dataset_name = 'flowers'

# Prepares flowers for inception
def define_training_parameters():
  
  file_mngr = cnn_files()
  file_mngr.get_or_init_training_set()
  define_training_parameters(file_mngr, dataset_name)  

# Prepares evaluation parameters
def define_eval_parameters():
  file_mngr = cnn_files()
  define_eval_parameters(file_mngr, dataset_name)