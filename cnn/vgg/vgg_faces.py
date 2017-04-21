"""
Created on Jan 30, 2017

Implementation of VGGFaces in Keras library

VGGFace model for Keras.
# Reference:
- [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model


hidden_dim = 512

TH_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v1.0/rcmalli_vggface_th_weights_th_ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v1.0/rcmalli_vggface_tf_weights_tf_ordering.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v1.0/rcmalli_vggface_th_weights_th_ordering_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v1.0/rcmalli_vggface_tf_weights_tf_ordering_notop.h5'

def _get_input_shape(include_top):
  """Gets input shape
    Args:
      include_top - include top layers
    Returns:
      input_shape - input tensor's shape
  """
  
  if include_top:
    input_shape = (224, 224, 3)
  else:
    input_shape = (None, None, 3)
      
  return input_shape

def _convert_tensor(input_tensor):
  """Converts input to Keras tensor
    Args:
      input_tensor - input tensor shape
    Returns:
      img_input - input image tensor      
  """

  if not K.is_keras_tensor(input_tensor):
      img_input = Input(tensor=input_tensor)
  else:
      img_input = input_tensor
  
  return img_input

def _get_image_input(include_top, input_tensor):
  """Initializes input image tensor
    Args:
      include_top - include top layers 
      input_tensor - input tensor shape
    Returns:
      img_input - input image tensor    
  """
  
  input_shape = _get_input_shape(include_top)
  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    img_input = _convert_tensor(input_tensor)
  
  return  img_input

def _load_weights(include_top, weights, model):
  """Loads weighs
    Args:
      include_top - include top layers
      weights - weights description
      model - network model      
  """

  if weights == 'vggface':
    print('K.image_dim_ordering:', K.image_dim_ordering())
    if K.image_dim_ordering() == 'th':
      if include_top:
        weights_path = get_file('rcmalli_vggface_th_weights_th_ordering.h5',
                                TH_WEIGHTS_PATH,
                                cache_subdir='models')
      else:
        weights_path = get_file('rcmalli_vggface_th_weights_th_ordering_notop.h5',
                                TH_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
      model.load_weights(weights_path)
      if K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                      'are using the Theano '
                      'image dimension ordering convention '
                      '(`image_dim_ordering="th"`). '
                      'For best performance, set '
                      '`image_dim_ordering="tf"` in '
                      'your Keras config '
                      'at ~/.keras/keras.json.')
        convert_all_kernels_in_model(model)
    else:
      if include_top:
        weights_path = get_file('rcmalli_vggface_tf_weights_tf_ordering.h5',
                                TF_WEIGHTS_PATH,
                                cache_subdir='models')
      else:
        weights_path = get_file('rcmalli_vggface_tf_weights_tf_ordering_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
      model.load_weights(weights_path)
      if K.backend() == 'theano':
        convert_all_kernels_in_model(model)
        
def network_model(flags):
  """Initializes last layers of model
    Args:
      flags - command line arguments
    Returns:
      model - network model
  """
  if flags.input_tensor:
    image_input = Input(shape=(224, 224, 3))
    vgg_model = VGGFace(input_tensor=image_input, include_top=False)
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    out = Dense(flags.nb_class, activation='softmax', name='fc8')(x)
    model = Model(image_input, out)
  else:
    model = VGGFace(include_top=flags.include_top)
    
  return model
                
def VGGFace(include_top=True, weights='vggface',
          input_tensor=None):
  
  """Instantiate the VGGFace architecture,
  optionally loading weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_dim_ordering="tf"` in your Keras config
  at ~/.keras/keras.json.
  The model and the weights are compatible with both
  TensorFlow and Theano. The dimension ordering
  convention used by the model is the one
  specified in your Keras config file.
  # Arguments
      include_top: whether to include the 3 fully-connected
          layers at the top of the network.
      weights: one of `None` (random initialization)
          or "vggface" (pre-training on VGGFace dataset).
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
  # Returns
      A Keras model instance.
  """
  if weights not in {'vggface', None}:
    raise ValueError('The `weights` argument should be either '
                      '`None` (random initialization) or `vggface` '
                      '(pre-training on VGGFace dataset).')
  
  img_input = _get_image_input(include_top, input_tensor)
  # Determine proper input shape
  # Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

  if include_top:
      # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(2622, activation='softmax', name='fc8')(x)

  # Create model
  model = Model(img_input, x)
  _load_weights(include_top, weights, model)

  return model
