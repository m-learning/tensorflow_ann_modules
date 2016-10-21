'''
Created on Oct 21, 2016
Logs events to TensorBoard
@author: Levan Tsinadze
'''

import os
from tempfile import gettempdir

import cnn.transfer.training_flags_mod as training_flags_mod
import tensorflow as tf


def init_writer(sess):
  """Initialized training board logger
    Args:
      sess - TensorFlow session
    Returns:
      merged - summaries merger
      train_writer - train log writer
      validation_writer - validation log writer
  """
  
  # Merge all the summaries and write them out to /tmp/retrain_inception_logs (by default)
  summaries_dir = os.path.join(gettempdir(), training_flags_mod.summaries_dir)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(summaries_dir + '/train',
                                        sess.graph)
  validation_writer = tf.train.SummaryWriter(summaries_dir + '/validation')
  
  return (merged, train_writer, validation_writer)
    