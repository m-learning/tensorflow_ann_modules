"""
Created on Jan 9, 2017

Face embeddings generator from FaceNet pre-trained model 

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import sys

import cnn.faces.facenet as facenet
import cnn.faces.lfw as lfw
import numpy as np
import tensorflow as tf

INPUT_LAYER = 'input:0'
EMBEDDINGS_LAYER = 'embeddings:0'

def image_to_embedding(sess, model_dir, mete_file, chp_file, paths):
  
  embs = []
  
  with tf.Graph().as_default():
    
      with tf.Session() as sess:
          
          # Load the model
          print('Model directory: %s' % model_dir)
          meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
          print('Metagraph file: %s' % meta_file)
          print('Checkpoint file: %s' % ckpt_file)
          facenet.load_model(model_dir, meta_file, ckpt_file)
          
          # Get input and output tensors
          images_placeholder = tf.get_default_graph().get_tensor_by_name(INPUT_LAYER)
          embeddings = tf.get_default_graph().get_tensor_by_name(EMBEDDINGS_LAYER)
          
          image_size = images_placeholder.get_shape()[1]
      
          # Run forward pass to calculate embeddings
          print('Runnning forward pass to generate embeddings')
          nrof_images = len(paths)
          nrof_batches = int(math.ceil(1.0 * nrof_images))
          for i in range(nrof_batches):
              start_index = i
              end_index = min((i + 1), nrof_images)
              paths_batch = paths[start_index:end_index]
              images = facenet.load_data(paths_batch, False, False, image_size)
              feed_dict = { images_placeholder:images }
              emb = sess.run(embeddings, feed_dict=feed_dict)
              embs.append(emb)
              
  return embs

def generate_embedding(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name(INPUT_LAYER)
            embeddings = tf.get_default_graph().get_tensor_by_name(EMBEDDINGS_LAYER)
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images }
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                args.seed, actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            
            facenet.plot_roc(fpr, tpr, 'NN4')
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=667)
    return parser.parse_args(argv)

if __name__ == '__main__':
    generate_embedding(parse_arguments(sys.argv[1:]))
