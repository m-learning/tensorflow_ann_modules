"""
Created on Feb 14, 2017

VGG-Faces modules for Keras library

@author: Levan Tsinadze
"""

from PIL import Image
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model

from cnn.faces.cnn_files import training_file
import numpy as np


WEIGHTS_FILE = 'vgg-face-keras-fc.h5'

def _weights_path():
  """Path for pre-trained weights
    Returns:
      _weights - pre-trained weights path
  """
  
  _files = training_file()
  _weights = _files.model_file(WEIGHTS_FILE)
  
  return _weights

def vgg_face(weights_path=None):
  
  img = Input(shape=(224, 224, 3))

  pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
  conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
  pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
  conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
  pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

  pad2_1 = ZeroPadding2D((1, 1))(pool1)
  conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
  pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
  conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
  pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

  pad3_1 = ZeroPadding2D((1, 1))(pool2)
  conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
  pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
  conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
  pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
  conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
  pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

  pad4_1 = ZeroPadding2D((1, 1))(pool3)
  conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
  pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
  conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
  pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
  conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
  pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

  pad5_1 = ZeroPadding2D((1, 1))(pool4)
  conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
  pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
  conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
  pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
  conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
  pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

  flat = Flatten()(pool5)
  fc6 = Dense(4096, activation='relu', name='fc6')(flat)
  fc6_drop = Dropout(0.5)(fc6)
  fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
  fc7_drop = Dropout(0.5)(fc7)
  out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

  model = Model(input=img, output=out)

  if weights_path:
      model.load_weights(weights_path)

  return model

def _read_image():
  """Reads image from file
    Returns:
      im - image tensor from file
  """
  
  im = Image.open('A.J._Buckley.jpg')
  im = im.resize((224, 224))
  im = np.array(im).astype(np.float32)
#    im[:,:,0] -= 129.1863
#    im[:,:,1] -= 104.7624
#    im[:,:,2] -= 93.5940
  im = im.transpose((2, 0, 1))
  im = np.expand_dims(im, axis=0)
  
  return im  

if __name__ == "__main__":
  
  im = _read_image()

  # Test pretrained model
  _weights = _weights_path()
  model = vgg_face(_weights)
  out = model.predict(im)
  print(out[0][0])
    
