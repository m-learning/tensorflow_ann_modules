"""
Created on Jul 15, 2016

Configures parameters before retraining

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from  cnn.transfer import graph_config
import cnn.transfer.training_flags as flags
import tensorflow as tf


def weight_variable(shape):
  """Create a weight variable with appropriate initialization
    Args:
      shape - shape of weight tensor
    Returns:
      variable placeholder
  """
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial, name='final_weights')

def bias_variable(shape):
  """Create a bias variable with appropriate initialization
    Args:
      shape - shape of bias tensor
    Returns:
      variable placeholder
  """
  initial = tf.zeros(shape)
  return tf.Variable(initial, name='final_biases')

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor 
     (for TensorBoard visualization).
    Args:
      var - variable
      name - name of variable
  """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)
    
def network_layer(layer_params):
  """Generates fully connected network layer
    Args:
      layer_paras - layer parameters
    Returns:
      preactivations - pre activation tensor
      activations - activation tensor
      keep_prob - placeholder for "dropout" keep propability parameter
  """
  (input_tensor, input_dim, output_dim, layer_name) = layer_params
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      layer_weights = weight_variable([input_dim, output_dim])
      variable_summaries(layer_weights, layer_name + '/weights')
    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.scalar_summary('dropout_keep_probability', keep_prob)
      drop = tf.nn.dropout(input_tensor, keep_prob)
      variable_summaries(drop, layer_name + '/dropout')    
    with tf.name_scope('biases'):
      layer_biases = bias_variable([output_dim])
      variable_summaries(layer_biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivations = tf.matmul(drop, layer_weights) + layer_biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivations)
  
  return (preactivations, keep_prob)
    
def full_network_layer(full_layer_params):
  """Generates fully connected network layer
    Args:
      layer_paras - layer parameters (dimensions, activation function etc)
    Returns:
      preactivations - pre activation tensor
      activations - activation tensor
      keep_prob - placeholder for "dropout" keep propability parameter
  """
  (input_tensor, input_dim, output_dim, layer_name,
   activation_name, activation_function) = full_layer_params
  layer_params = (input_tensor, input_dim, output_dim, layer_name)
  (preactivations, keep_prob) = network_layer(layer_params)
  activations = activation_function(preactivations, name=activation_name)
  tf.histogram_summary(activation_name + '/activations', activations)
  
  return (preactivations, activations, keep_prob)

def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input and dropout keep probability.
  """
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, graph_config.BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32, [None, class_count],
                                        name='GroundTruthInput')
  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  layer_params = (bottleneck_input, graph_config.BOTTLENECK_TENSOR_SIZE,
                  class_count, layer_name, final_tensor_name, tf.nn.softmax)
  (logits, final_tensor, keep_prob) = full_network_layer(layer_params)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, ground_truth_input)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.scalar_summary('cross entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(
      flags.learning_rate).minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input,
          ground_truth_input, final_tensor, keep_prob)

def init_flags_only(tr_file):
  """Configures trained checkpoints
    Args:
      tr_file - utility for files management
  """
  flags.init_flaged_data(tr_file)

def init_flags_and_files(tr_file):
  """Initializes training flags
    Args:
      tr_file - file utility manager
  """
  
  init_flags_only(tr_file)
  tr_file.get_or_init_training_set()
