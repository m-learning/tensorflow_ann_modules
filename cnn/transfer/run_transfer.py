"""
Created on Nov 13, 2016
Runs transfer learning for Inception-V3 model on custom data set
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.transfer import retrain_network as retrainer
from cnn.transfer import training_flags as flags
from cnn.transfer.cnn_files import training_file
import tensorflow as tf


def retrain_net_main(_):
  """Retrains Inception-V3 on custom data set"""
  
  tr_files = training_file()
  flags.parse_and_retrieve(tr_files)
  retrainer.retrain_net(tr_files)
  
if __name__ == '__main__':
  """Runs training iterations and test process"""
  tf.app.run(retrain_net_main)
