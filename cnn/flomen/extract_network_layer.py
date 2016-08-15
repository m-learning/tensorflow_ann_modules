'''
Created on Aug 15, 2016

Extracts pool info

@author: Levan Tsinadze
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from cnn.transfer.extract_layer import layer_features
from cnn.flomen.cnn_files import training_file

# Training and testing
def extract_net_main(_):
  
  tr_files = training_file()
  features = layer_features('pool_3:0')
  features.extract_layer(tr_files)
  
  
# Runs layer extractor
if __name__ == '__main__':
  tf.app.run(extract_net_main)