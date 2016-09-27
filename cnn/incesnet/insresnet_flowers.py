# '''
# Created on Sep 23, 2016
#
# Retraining of inception-resnet for flowers data set
#
# @author: Levan tsinadze
# '''

import sys

from cnn.flowers.cnn_files import training_file
from cnn.incesnet.config_parameters import train_and_eval_config


dataset_name = 'flowers'

class flower_config(train_and_eval_config):
  
  def __init__(self):
    super(flower_config, self).__init__(training_file(), dataset_name)

if __name__ == '__main__':
  flomenn_cfg = flower_config()
  flomenn_cfg.train_or_eval_net(sys.argv)
