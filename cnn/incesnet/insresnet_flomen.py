# '''
# Created on Sep 23, 2016
#
# Retraining of inception-resnet for flowers data set
#
# @author: Levan Tsinadze
# '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from cnn.datasets import download_and_convert_flomen
from cnn.flomen.cnn_files import training_file
from cnn.incesnet.config_parameters import train_and_eval_config

# Data set name
dataset_name = 'flomen'

# Configuration for flomen data set
class flomen_config(train_and_eval_config):
  
  def __init__(self):
    super(flomen_config, self).__init__(training_file(), dataset_name,
                                        download_and_convert_flomen)

if __name__ == '__main__':
  flomenn_cfg = flomen_config()
  flomenn_cfg.train_or_eval_net(sys.argv)
