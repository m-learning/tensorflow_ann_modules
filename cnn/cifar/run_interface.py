"""
Created on Dec 27, 2016

Runs CIFAR10 network interface for image recognition

@author: Levan Tsinadze
"""

import os

from cnn.cifar import argument_reader as reader
from cnn.cifar import eval_cifar as evaluator
from cnn.cifar import input_cifar as inputs
from cnn.cifar import network_config as network
import tensorflow as tf


FLAGS = None

def _get_file_names():
  """Gets file names for recognition 
    Returns:
      filenames - collection of file paths
  """

  if os.path.isdir(FLAGS.file_path):
    filenames = [os.path.join(FLAGS.file_path, f) for f in os.listdir(FLAGS.file_path)]
    print(filenames)
  else:
    filenames = [FLAGS.file_path]
    
  return filenames

def eval_interface(argsv=None):
  """Evaluates CIFAR network interface for instant file
    Args:
      file_path - path for file or 
                  directory of files for recognition
    Returns:
      answer - recognition result
  """
  if os.path.isdir(FLAGS.file_path):
    filenames = [os.path.join(FLAGS.file_path, f) for f in os.listdir(FLAGS.file_path)]
    print(filenames)
  else:
    filenames = [FLAGS.file_path]
  
  with tf.Graph().as_default() as g:
  
    (images, labels) = inputs.input_from_filenames(filenames)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = network.inference(images)
  
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
  
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
  
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
  
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    
    evaluator.eval_once(saver, summary_writer, top_k_op, summary_op)

def parse_and_retrieve():
  """Parses command line arguments"""
  
  global FLAGS
  FLAGS = reader.parse_and_retrieve(batch_size=1, num_examples=1)
  evaluator.FLAGS = FLAGS

if __name__ == '__main__':
  parse_and_retrieve()
  tf.app.run(eval_interface)
