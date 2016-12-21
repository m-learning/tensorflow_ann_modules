"""
Created on Jul 8, 2016

A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/


@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import time

from cnn.cifar import network_config as network 
from cnn.cifar.cnn_files import training_file
import tensorflow as tf


FLAGS = None

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = network.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = network.inference(images)

    # Calculate loss.
    loss = network.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = network.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time
        loss_value = run_values.results
        if self._step % 10 == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def prepare_and_train(argv=None):  # pylint: disable=unused-argument
  network.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

def parse_and_retrieve():
  """Parses command line arguments"""
  
  __files = training_file()
  global FLAGS
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--train_dir',
                          type=str,
                          default=__files.init_files_directory(),
                          help='Directory where to write event logs and checkpoint.')
  arg_parser.add_argument('--max_steps',
                          type=int,
                          default=1000000,
                          help='Number of batches to run.')
  arg_parser.add_argument('--log_device_placement',
                          dest='log_device_placement',
                          action='store_true',
                          help='Whether to log device placement.')
  arg_parser.add_argument('--not_log_device_placement',
                          dest='log_device_placement',
                          action='store_false',
                          help='Whether to log device placement.')
  arg_parser.add_argument('--batch_size',
                          type=int,
                          default=128,
                          help='Number of images to process in a batch.')
  arg_parser.add_argument('--data_dir',
                          type=str,
                          default=__files.get_training_directory(),
                          help='Path to the CIFAR-10 data directory.')
  arg_parser.add_argument('--log_files',
                          type=str,
                          default=__files.init_log_files(),
                          help='Path to the training output log files.')
  arg_parser.add_argument('--log_errors',
                          type=str,
                          default=__files.init_error_files(),
                          help='Path to the training error log files.')
  arg_parser.add_argument('--use_fp16',
                          dest='use_fp16',
                          action='store_true',
                          help='Train the model using fp16.')
  arg_parser.add_argument('--not_use_fp16',
                          dest='use_fp16',
                          action='store_false',
                          help='Train the model using fp32.')
  (FLAGS, _) = arg_parser.parse_known_args()
  print('parameters identified:')
  print('train_dir', FLAGS.train_dir)
  print('data_dir', FLAGS.data_dir)
  print('log_files', FLAGS.log_files)
  print('log_errors', FLAGS.log_errors)
  print('log_device_placement', FLAGS.log_device_placement)
  print('use_fp16', FLAGS.use_fp16)
  network.FLAGS = FLAGS

if __name__ == '__main__':
  parse_and_retrieve()
  tf.app.run(prepare_and_train)
