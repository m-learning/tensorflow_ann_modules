"""
Created on Jul 15, 2016

Configuration for neural network graph

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import traceback

from tensorflow.python.platform import gfile

import cnn.transfer.training_flags as flags
import tensorflow as tf


# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_FEATURE_TENSOR_NAME = 'pool_3:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299  # Input image width
MODEL_INPUT_HEIGHT = 299  # Input image height
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

# Final layer name
FINAL_LAYER_NAME = 'final_training_ops'

def init_model_file_name():
  """Initializes serialized model graph file 
    Returns:
      file full path
  """
  return os.path.join(flags.model_dir, 'classify_image_graph_def.pb')

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
  """
  with tf.Session() as sess:
    model_filename = init_model_file_name()
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
                              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                              RESIZED_INPUT_TENSOR_NAME]))
  
    return (sess, sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor)

def list_layer_values(values, layer_name):
  """List all network layers
    Args:
      values - layers to filter
      layer_name - name of layer to list
    Returns:
      result - retrieved layer by key
  """
  
  result = None
  
  for value in values:
    print(value.name)
    if value.name == layer_name:
      result = value
    
  return result

def list_from_layer_op(layer_op, layer_name):
  """Lists all network layers
    Args:
      layer_op - layer operations
      layer_name - layer name
    Returns:
      result - retrieved layer by key
  """
  values = layer_op.values
  result = list_layer_values(values(), layer_name)
  
  return result
  
def list_layers(sess, layer_name):
  """List All network layers
    Args:
      sess - current TensorFlow session
      layer_name - name of layer to list
    Returns:
      result - retrieved layer by key
  """
  result = None
  
  layer_ops = sess.graph.get_operations()
  for layer_op in layer_ops:
    result = list_from_layer_op(layer_op, layer_name)
    if result is not None:
      break
  
  return result
  
def get_layer(layer_name):  
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Args:
    layer_name - network layer name
  Returns:
    net_layer - Graph holding the trained Inception network, 
                and various tensors we'll be manipulating.
  """
  
  with tf.Session() as sess:
    model_filename = init_model_file_name()
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def)
      try:
        net_layer = list_layers(sess, layer_name)
      except Exception:
        net_layer = None
        print('Error occurred')
        traceback.print_exc()
  
  return net_layer
