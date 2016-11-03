"""
Created on Jul 15, 2016

Configuration for neural network graph

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from tensorflow.python.platform import gfile
import traceback

import cnn.transfer.training_flags as flags
import tensorflow as tf


# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299  # Input image width
MODEL_INPUT_HEIGHT = 299  # Input image height
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

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
  
  # Graph components
  return (sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor)

def list_layer_values(values, layer_name):
  """List All network layers
    Args:
      values - layers to filter
      layer_name - name of layer to list
    Returns:
      result - retrieved layer by key
  """
  
  result = None
  
  for value in values:
    print(type(value))
    print(value._op.name)
    print(value.name)
    if value.name == layer_name:
      result = value
    print(value)
    
  return result

def list_layers(sess, layer_name):
  """List All network layers
    Args:
      sess - TensorFlow session
      layer_name - name of layer to list
    Returns:
      result - retrieved layer by key
  """
  
  result = None
  
  layer_ops = sess.graph.get_operations()
  print(layer_ops)
  for layer_op in layer_ops:
    values = layer_op.values()
    result = list_layer_values(values, layer_name)
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
  
  net_layer = None
  
  with tf.Session() as sess:
    model_filename = init_model_file_name()
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def)
      try:
        net_layer = list_layers(sess, layer_name)
        print(net_layer)
      except Exception:
        print('Error occurred')
        traceback.print_exc()
  
  return net_layer
        
