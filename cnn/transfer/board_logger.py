"""
Created on Oct 21, 2016

Logs events to TensorBoard

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tempfile import gettempdir

import cnn.transfer.training_flags as flags
import tensorflow as tf


def init_log_directories(sess):
  """Initializes training and validation board 
     logger directories
    Args:
      sess - current TensorFlow session
    Returns:
      training_sum_dir - training summaries directory
      validatn_sum_dir - validation summaries directory
  """
  tmp_dir = gettempdir()
  summaries_dir = os.path.join(tmp_dir, flags.summaries_dir)
  if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
  tf.gfile.MakeDirs(summaries_dir)
  training_sum_dir = summaries_dir + '/train'
  validatn_sum_dir = summaries_dir + '/validation'
  
  return (training_sum_dir, validatn_sum_dir)

def init_writer(sess):
  """Initialized training and validation board logger
    Args:
      sess - current TensorFlow session
    Returns:
      merged - summaries merger
      train_writer - train log writer
      validation_writer - validation log writer
  """
  (training_sum_dir, validatn_sum_dir) = init_log_directories(sess)
  # Merge all the summaries and write them out to /tmp/retrain_inception_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(training_sum_dir, sess.graph)
  validation_writer = tf.summary.FileWriter(validatn_sum_dir)
  
  return (merged, train_writer, validation_writer)
    
