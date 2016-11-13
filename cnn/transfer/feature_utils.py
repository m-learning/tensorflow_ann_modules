"""
Created on Nov 10, 2016
Features extractor from image
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile

from cnn.transfer import graph_config as gconf
import numpy as np
import tensorflow as tf


def extract_features(sess, flattened_tensor, image_data):
  """Extracts features from passed image data
    Args:
      sess - current TensorFlow session
      flattened_tensor - pooling leayer
      image_data - binary image
    Returns:
      features - extracted features
  """
  
  feature = sess.run(flattened_tensor, {
            gconf.JPEG_DATA_TENSOR_NAME: image_data
        })
  features = np.squeeze(feature)
  
  return features
  
def feature_extractor(image_paths, verbose=True):
  """Extracts features by Inception-V3 from image
    Args:
      image_paths - image paths to extract features
      verbose - logging utilities
    Returns:
      features - extracted features from images
  """

  features = np.empty((len(image_paths), gconf.BOTTLENECK_TENSOR_SIZE))
  with tf.Session() as sess:
    flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    for i, image_path in enumerate(image_paths):
        if verbose:
            print('Processing %s...' % (image_path))
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        features[i, :] = extract_features(sess, flattened_tensor, image_data)

    return features 