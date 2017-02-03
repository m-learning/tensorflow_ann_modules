"""
Created on Feb 1, 2017

Network model configuration

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rnn.wlm import reader
import tensorflow as tf


class ModelInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    produce_data = (data, batch_size, num_steps)
    self.input_data, self.targets = reader.data_producer(produce_data, name=name)

class SmallConfig(object):
  """Small config."""
  
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  data_type = tf.float32


class MediumConfig(object):
  """Medium configuration."""
  
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  data_type = tf.float32


class LargeConfig(object):
  """Large configuration."""
  
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  data_type = tf.float32


class TestConfig(object):
  """Tiny configuration, for testing."""
  
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  data_type = tf.float32

def data_type(FLAGS):
  """Gets float point type from configuration
    Returns:
      float point type
  """
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def _init_config(FLAGS):
  """Initializes network configuration
    Args:
      FLAGS = training flags
    Returns:
      conf = network configuration
  """
  
  if FLAGS.model == "small":
    conf = SmallConfig()
  elif FLAGS.model == "medium":
    conf = MediumConfig()
  elif FLAGS.model == "large":
    conf = LargeConfig()
  elif FLAGS.model == "test":
    conf = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  
  return conf

def get_config(FLAGS):
  """Initializes and gets network model configuration
    Args:
      FLAGS = training flags
    Returns:
      conf = network configuration
  """
  
  conf = _init_config(FLAGS)
  conf.data_type = data_type(FLAGS)
  
  return conf
