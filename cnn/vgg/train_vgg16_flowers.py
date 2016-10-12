# '''
# Created on Sep 23, 2016
#
# Retraining of inception-resnet for flowers data set
#
# @author: Levan tsinadze
# '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from cnn.datasets import download_and_convert_flowers
from cnn.flowers.cnn_files import training_file
from cnn.nets.config_parameters import train_and_eval_config
import cnn.vgg.vgg_constants as constants


# Data set name
dataset_name = 'flowers'

training_parameters = (training_file(), dataset_name,
                       download_and_convert_flowers,
                       None, None,
                       constants.checkpoint_file,
                       constants.checkpoint_url)

# Configuration for flowers data set
class flower_config(train_and_eval_config):
  
  def __init__(self):
    super(flower_config, self).__init__(training_parameters)
  
  # Addts configuration parameters and trains model
  def config_and_train(self, sys_args):
    """Configures and trains or evaluates
    Args:
    system arguments to decide between train and evaluate.
    global_step: The global_step tensor.
    """
    
    self.run_config_function(sys_args)
    self.set_model_name('vgg_16')
    self.set_trainable_and_exclude_scopes(constants.checkpoint_exclude_scopes,
                                          constants.trainable_scopes)
    self.set_optimizer('sgd')
    self.set_max_number_of_steps(4000)
    self.train_or_eval_net(sys_args)

if __name__ == '__main__':
  model_cfg = flower_config()
  model_cfg.config_and_train(sys.argv)
