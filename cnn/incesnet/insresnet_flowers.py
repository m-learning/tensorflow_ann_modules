# '''
# Created on Sep 23, 2016
#
# Retraining of inception-resnet for flowers data set
#
# @author: Levan tsinadze
# '''

import sys

from cnn.datasets import download_and_convert_flowers
from cnn.flowers.cnn_files import training_file
from cnn.incesnet.config_parameters import train_and_eval_config

# Data set name
dataset_name = 'flowers'

# Configuration for flowers data set
class flower_config(train_and_eval_config):
  
  def __init__(self):
    super(flower_config, self).__init__(training_file(), dataset_name, download_and_convert_flowers)

if __name__ == '__main__':
  flomenn_cfg = flower_config()
  flomenn_cfg.train_or_eval_net(sys.argv)
