"""
Created on Feb 1, 2017

Trains word language model LSTM network

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from rnn.wlm import reader
from rnn.wlm.model_config import get_config, \
                                 ModelInput
from rnn.wlm.network_model import NetworkModel
from rnn.wlm.rnn_files import training_file
import tensorflow as tf                                 


logging = tf.logging

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data.
    Returns:
      result - result from one epoch
  """
  
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" % 
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))
  result = np.exp(costs / iters)
  
  return result

def _train():
  """Runs training epochs and saves weights"""
  
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.read_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config(FLAGS)
  eval_config = get_config(FLAGS)
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
      train_input = ModelInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = NetworkModel(is_training=True, config=config, input_=train_input)
      tf.scalar_summary("Training Loss", m.cost)
      tf.scalar_summary("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = ModelInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = NetworkModel(is_training=False, config=config, input_=valid_input)
      tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = ModelInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = NetworkModel(is_training=False, config=eval_config, input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

def _parse_arguments():
  """Parses command line arguments"""
  
  global FLAGS
  _files = training_file()
  parser = argparse.ArgumentParser()
  parser.add_argument('--model',
                      type=str,
                      default='small',
                      help='A type of model. Possible options are: small, medium, large.')
  parser.add_argument('--data_path',
                      type=str,
                      default=_files.data_dir,
                      help='Where the training/test data is stored.')
  parser.add_argument('--save_path"',
                      type=str,
                      default=_files.model_dir,
                      help='Model output directory.')
  parser.add_argument('--use_fp16',
                      dest='use_fp16',
                      action='store_true',
                      help='Train using 16-bit floats instead of 32bit floats')
  (FLAGS, _) = parser.parse_known_args()
  
def config_and_train(_):
  
  """Configures (reads command line arguments) and trains network model"""
  _parse_arguments()
  _train()

if __name__ == "__main__":
  tf.app.run(config_and_train)
