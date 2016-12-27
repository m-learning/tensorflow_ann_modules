"""
Created on Dec 27, 2016

Runs CIFAR10 network interface for image recognition

@author: Levan Tsinadze
"""

import argparse
from datetime import datetime
import math
import os

from cnn.cifar import eval_cifar as evaluator
from cnn.cifar import input_cifar as inputs
from cnn.cifar import network_config as network
from cnn.cifar.cnn_files import training_file
import numpy as np
import tensorflow as tf


FLAGS = None

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
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
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def eval_interface(argsv=None):
  """Evaluates CIFAR network interface for instant file
    Args:
      file_path - path for file or 
                  directory of files for recognition
    Returns:
      answer - recognition result
  """
  if os.path.isdir(FLAGS.file_path):
    filenames = os.listdir(FLAGS.file_path)
    
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
  
  __files = training_file()
  __default_data_dir = __files.get_training_directory()
  global FLAGS
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--file_path',
                          type=str,
                          default=os.path.join(__default_data_dir, 'test_batch.bin'),
                          help='File or directory path for evaluation') 
  arg_parser.add_argument('--eval_dir',
                          type=str,
                          default=__files.init_logs_directory(),
                          help='Directory where to write event logs.')
  arg_parser.add_argument('--eval_data',
                          type=str,
                          default='test',
                          help='Either "test" or "train_eval".')
  arg_parser.add_argument('--checkpoint_dir',
                          type=str,
                          default=__files.init_files_directory(),
                          help='Directory where to read model checkpoints.')
  arg_parser.add_argument('--eval_interval_secs',
                          type=int,
                          default=60 * 5,
                          help='How often to run the eval.')
  arg_parser.add_argument('--num_examples',
                          type=int,
                          default=10000,
                          help='Number of examples to run.')
  arg_parser.add_argument('--run_once',
                          dest='run_once',
                          action='store_true',
                          help='Whether to run eval only once.')
  arg_parser.add_argument('--not_run_once',
                          dest='run_once',
                          action='store_false',
                          help='Whether to run eval not only once.')
  arg_parser.add_argument('--batch_size',
                          type=int,
                          default=128,
                          help='Number of images to process in a batch.')
  arg_parser.add_argument('--data_dir',
                          type=str,
                          default=__default_data_dir,
                          help='Path to the CIFAR-10 data directory.')
  arg_parser.add_argument('--use_fp16',
                          dest='use_fp16',
                          action='store_true',
                          help='Train the model using fp16.')
  arg_parser.add_argument('--not_use_fp16',
                          dest='use_fp16',
                          action='store_false',
                          help='Train the model using fp32.')
  (FLAGS, _) = arg_parser.parse_known_args()
  network.FLAGS = FLAGS

if __name__ == '__main__':
  parse_and_retrieve()
  tf.app.run(eval_interface)
