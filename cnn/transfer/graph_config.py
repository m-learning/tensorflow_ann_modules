'''
Created on Jul 15, 2016

Configuration for neural network graph

@author: Levan Tsinadze
'''

import os.path
import traceback

import tensorflow as tf
from tensorflow.python.platform import gfile

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

# Gets graph file
def init_model_file_name(tr_flags):
  """Initializes serialized model graph file 
    Args:
      tr_flags - training configuration
    Returns:
      file full path
  """
  return os.path.join(tr_flags.model_dir, 'classify_image_graph_def.pb')

# Generates neural network model graph
def create_inception_graph(tr_flags):
  """"Creates a graph from saved GraphDef file and returns a Graph object.
    Args:
      tr_flags - configuration flags
    Returns:
      Graph holding the trained Inception network, and various tensors we'll be
      manipulating.
  """
  with tf.Session() as sess:
    model_filename = init_model_file_name(tr_flags)
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  
  # Graph components
  return (sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor)

# List values of graph
def list_layer_values(values, layer_name):
  """List All network layers
    Args:
      values - layers to filter
      layer_name - name of layer to list
  """
  
  result = None
  
  for value in values:
    print type(value)
    print value._op.name
    print value.name
    if value.name == layer_name:
      result = value
    print value
    
  return result
  

# Lists all layers of network
def list_layers(sess, layer_name):
  """List All network layers
    Args:
      sess - TensorFlow session
      layer_name - name of layer to list
  """
  
  result = None
  
  layer_ops = sess.graph.get_operations()
  print layer_ops
  for layer_op in layer_ops:
    values = layer_op.values()
    result = list_layer_values(values, layer_name)
    if result is not None:
      break
  
  return result
  

# Gets network graph layer by name
def get_layer(tr_flags, layer_name):  
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Args:
    tr_flags - configuration parameters
    layer_name - network layer name
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = init_model_file_name(tr_flags)
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def)
      try:
        net_layer = list_layers(sess, layer_name)
        print net_layer
      except Exception:
        print 'Error occured'
        traceback.print_exc()
        