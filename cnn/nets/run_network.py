# '''
# Created on Oct 6, 2016
# General interface module for Inception-ResNet-v2 implementation
# @author: Levan Tsinadze
# '''

import cnn.vgg.vgg as vgg
import numpy as np
import tensorflow as tf


interface_function = vgg.vgg_16
network_name = 'vgg16'
layer_key = 'vgg_16/fc8'
endpoint_layer = 'fc8'

# Runs Inception-ResNet-v2 Module
class network_interface(object):
  
  def __init__(self, cnn_file, checkpoint_file=None):
    
    self.cnn_file = cnn_file
    self.checkpoint_path = cnn_file.init_files_directory()
    if checkpoint_file is None:
      self.checkpoint_dir = tf.train.latest_checkpoint(self.checkpoint_path)
    else:
      self.checkpoint_dir = self.cnn_file.join_path(self.checkpoint_path, checkpoint_file)
  
  # Generates labels file
  def generate_labels(self):
    
    labels_path = self.cnn_file.join_path(self.cnn_file.get_dataset_dir, 'labels.txt')
    f = open(labels_path, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    
    return labels
  
  # Prints predicted answer
  def print_answer(self, predict_values):
    
    predictions = np.squeeze(predict_values)
    top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
    labels = self.generate_labels()
    for node_id in top_k:
      print node_id
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
