"""
Created on Jan 12, 2017

Utility module for FaceNet model

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from scipy import misc

from cnn.faces import detect_face, facenet
import numpy as np
import tensorflow as tf


GRAPH_FILE = 'face_embeddings.pb'

INPUT_NAME = 'input'

INPUT_LAYER = 'input:0'
TRAIN_LAYER = 'phase_train:0'
EMBEDDINGS_LAYER = 'embeddings:0'

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction, _files):
  """Loads and alighn face images from files
    Args:
      image_paths - image file paths
      image_size - image size
      margin - margin for alignment
      gpu_memory_fraction - GPU memory fraction for parallel processing
    Returns:
      images - aligned images from files
  """

  minsize = 20  # minimum size of face
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
  factor = 0.709  # scale factor
  
  print('Creating networks and loading parameters')
  with tf.Graph().as_default() as g:
    sess = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, _files.model_dir)

  nrof_samples = len(image_paths)
  img_list = [None] * nrof_samples
  for i in xrange(nrof_samples):
    img = misc.imread(os.path.expanduser(image_paths[i]))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    img_list[i] = prewhitened
  images = np.stack(img_list)
  
  return images