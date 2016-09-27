# '''
# Created on Sep 23, 2016
#
# Retraining of inception-resnet for flowers data set
#
# @author: Levan Tsinadze
# '''

import sys

from cnn.flomen.cnn_files import training_file
from cnn.incesnet.config_parameters import train_and_eval_config


dataset_name = 'flomen'

class flomen_config(train_and_eval_config):
  
  def __init__(self):
    super(flomen_config, self).__init__(training_file(), dataset_name)

if __name__ == '__main__':
  flomenn_cfg = flomen_config()
  flomenn_cfg.train_or_eval_net(sys.argv)
