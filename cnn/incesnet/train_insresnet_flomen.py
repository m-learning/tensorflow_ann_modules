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
import cnn.incesnet.evaluate_inception_resnet_v2.run_evaluation as eval_inception
import cnn.incesnet.retrain_inception_resnet_v2.run_training as train_inception
from cnn.nets.config_parameters import train_and_eval_config


# Data set name
dataset_name = 'flomen'

training_parameters = (training_file(), dataset_name,
                       download_and_convert_flomen,
                       train_inception, eval_inception,
                       'inception_resnet_v2_2016_08_30',
                       'inception_resnet_v2_2016_08_30')

# Configuration for flomen data set
class flomen_config(train_and_eval_config):
  
  def __init__(self):
    super(flomen_config, self).__init__(training_parameters)

  # Addts configuration parameters and trains model
  def config_and_train(self, sys_args):
    
    self.run_config_function(sys_args)
    self.set_model_name('inception_resnet_v2')
    self.set_trainable_and_exclude_scopes('InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits',
                                          'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits')
    self.set_max_number_of_steps(6000)
    self.set_learning_rate_decay_type('exponential')
    self.train_or_eval_net(sys_args)

if __name__ == '__main__':
  model_cfg = flomen_config()
  model_cfg.config_and_train(sys.argv)
