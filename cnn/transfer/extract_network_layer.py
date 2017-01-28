'''
Created on Aug 15, 2016

Extracts pool info

@author: Levan Tsinadze
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.transfer.cnn_files import training_file
from cnn.transfer.layer_extractor import layer_features
import tensorflow as tf


def extract_net_main(_):
  """Extracts features network layer"""
  
  tr_files = training_file()
  features = layer_features('pool_3:0')
  features.extract_layer(tr_files)
  
  
if __name__ == '__main__':
  """Runs layer extractor"""
  tf.app.run(extract_net_main)
