'''
Created on Sep 23, 2016

Retraining of inception-resnet for flowers data set

@author: Levan tsinadze
'''
from cnn.flowers.cnn_files import training_file
from cnn.incesnet.config_parameters import define_training_parameters, define_eval_parameters
import cnn.incesnet.retrain_inception_resnet_v2 as train_inception
import cnn.incesnet.evaluate_inception_resnet_v2 as eval_inception
import sys

dataset_name = 'flowers'

# Prepares flowers for inception
def init_training_parameters():
  
  file_mngr = training_file()
  file_mngr.get_or_init_training_set()
  define_training_parameters(file_mngr, dataset_name)  

# Prepares evaluation parameters
def init_eval_parameters():
  file_mngr = training_file()
  define_eval_parameters(file_mngr, dataset_name)
  
# Trains network
def train_net():
  init_training_parameters()
  train_inception.train_net()

# Evaluates network  
def eval_net():
  init_eval_parameters()
  eval_inception.eval_net()
  
if __name__ == '__main__':
  
  if len(sys.argv) > 1:
    if sys.argv[1] == 'eval':
      eval_net()
    else:
      train_net()
  else:
    train_net()