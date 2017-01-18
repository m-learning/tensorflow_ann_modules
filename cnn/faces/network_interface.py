"""
Created on Jan 18, 2017

Network interface for embeddings generation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cnn.faces import facenet
from cnn.faces.face_utils import EMBEDDINGS_LAYER, INPUT_LAYER
import tensorflow as tf


def calculate_embeddings_with_graph(sess, images):
  """Calculates embeddings for images
    Args:
      sess - current TensorFlow session
      images - image files
    Returns:
      emb - embeddings for images
  """  
  
  # Get input and output tensors
  images_placeholder = tf.get_default_graph().get_tensor_by_name(INPUT_LAYER)
  embeddings = tf.get_default_graph().get_tensor_by_name(EMBEDDINGS_LAYER)

  # Run forward pass to calculate embeddings
  feed_dict = { images_placeholder: images }
  emb = sess.run(embeddings, feed_dict=feed_dict)
  
  return emb

def calculate_embeddings(model_dir, images):
  """Calculates embeddings for images
    Args:
      model_dir - model directory
      images - image files
    Returns:
      emb - embeddings for images
  """
  
  with tf.Graph().as_default() as g:
    with tf.Session(graph=g) as sess:
      # Load the model
      print('Model directory: %s' % model_dir)
      meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
      print('Metagraph file: %s' % meta_file)
      print('Checkpoint file: %s' % ckpt_file)
      facenet.load_model(model_dir, meta_file, ckpt_file)
      emb = calculate_embeddings_with_graph(sess, images)
      
      return emb
