"""
Created on Aug 27, 2017

Utility module to download and convert data-set files

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import struct

from keras.utils.data_utils import get_file

import numpy as np


TR_FILE = 'train-images-idx3-ubyte.gz'
TR_LABELS = 'train-labels-idx1-ubyte.gz'
TS_FILE = 't10k-images-idx3-ubyte.gz'
TS_LABELS = 't10k-labels-idx1-ubyte.gz'

TRAIN_PATH = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + TR_FILE
TRAIN_LABELS = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + TR_LABELS
TEST_PATH = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + TS_FILE
TEST_LABELS = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + TS_LABELS

def _read_data(img_path, lbl_path):
  """Reads images in "numpy" format
    Args:
      img_path - path to images
      lbl_path - path to labels
    Returns:
      tuple of -
        images - image array
        labels - label array
  """

  with gzip.open(lbl_path, 'rb') as lbpath:
    struct.unpack('>II', lbpath.read(8))
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

  with gzip.open(img_path, 'rb') as imgpath:
    struct.unpack(">IIII", imgpath.read(16))
    images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

  return (images, labels)


def load_data():
  """Initialize data
    Returns:
      tuple of -
        x_train - training set
        y_train training labels
        x_test - testing set
        y_test - testing labels      
  """
  
  tr_file_path = get_file(TR_FILE, origin=TRAIN_PATH)
  tr_labl_path = get_file(TR_LABELS, origin=TRAIN_LABELS)
  ts_file_path = get_file(TS_FILE, origin=TEST_PATH)
  ts_labl_path = get_file(TS_LABELS, origin=TEST_LABELS)
  
  (tr_images, tr_labels) = _read_data(tr_file_path, tr_labl_path)
  (ts_images, ts_labels) = _read_data(ts_file_path, ts_labl_path)
  
  return (tr_images, tr_labels, ts_images, ts_labels)
  
  

  
