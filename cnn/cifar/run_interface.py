"""
Created on Dec 27, 2016

Runs CIFAR10 network interface for image recognition

@author: Levan Tsinadze
"""

import math
import os

from cnn.cifar import argument_reader as reader
from cnn.cifar import eval_cifar as evaluator
from cnn.cifar import input_cifar as inputs
from cnn.cifar import network_config as network
import numpy as np
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

def eval_once(saver, logits_labels):
  """Run Eval once.

  Args:
    logits_labels - tuple of -
      logits and labels
  """
  (logits, labels, filename) = logits_labels
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      step = 0
      while step < num_iter and not coord.should_stop():
        nlogits = sess.run(logits)
        preds = np.squeeze(nlogits)
        top_k = preds.argsort()[-5:][::-1]
        print(filename, ' - ')
        for node_id in top_k:
          human_string = labels[node_id]
          score = preds[node_id]
          print('%s (score = %.5f)' % (human_string, score))
        step += 1

      # Compute precision @ 1.

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  
def eval_for_file(filename):
  """Evaluates CIFAR network interface for instant file
    Args:
      filename - path for file 
    Returns:
      answer - recognition result
  """
  
  filenames = [filename]
  
  with tf.Graph().as_default():
    
    (_, images) = inputs.read_image_file(filenames)
    image_batch = tf.train.batch([images], FLAGS.batch_size)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = network.inference(image_batch)
    labels_path = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin', 'batches.meta.txt')
    f = open(labels_path, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    soft_max = tf.nn.softmax(logits)
    # Calculate predictions.
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
  
    
    eval_once(saver, logits_labels=(soft_max, labels, filename))

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
    for filename in filenames:
      eval_for_file(filename)
  else:
    filename = [FLAGS.file_path]
    eval_for_file(filename)

def parse_and_retrieve():
  """Parses command line arguments"""
  
  global FLAGS
  FLAGS = reader.parse_and_retrieve(batch_size=1, num_examples=1)
  evaluator.FLAGS = FLAGS

if __name__ == '__main__':
  parse_and_retrieve()
  tf.app.run(eval_interface)
