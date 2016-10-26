'''
Created on Oct 21, 2016
Logs events to TensorBoard
@author: Levan Tsinadze
'''

import os
from tempfile import gettempdir

import cnn.transfer.training_flags_mod as flags
import tensorflow as tf


def init_log_directories(sess):
  """Initializes training and validation board 
     logger directories
    Args:
      sess - TensorFlow session
    Return:
      training_sum_dir - training summaries directory
      validatn_sum_dir - validation summaries directory
  """
  tmp_dir = gettempdir()
  summaries_dir = os.path.join(tmp_dir, flags.summaries_dir)
  training_sum_dir = summaries_dir + '/train'
  validatn_sum_dir = summaries_dir + '/validation'
  
  return (training_sum_dir, validatn_sum_dir)

def init_writer(sess):
  """Initialized training and validation board logger
    Args:
      sess - TensorFlow session
    Return:
      merged - summaries merger
      train_writer - train log writer
      validation_writer - validation log writer
  """
  (training_sum_dir, validatn_sum_dir) = init_log_directories(sess)
  # Merge all the summaries and write them out to /tmp/retrain_inception_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(training_sum_dir, sess.graph)
  validation_writer = tf.train.SummaryWriter(validatn_sum_dir)
  
  return (merged, train_writer, validation_writer)
    
