"""
Created on Oct 17, 2016
Test class to preprocess images
@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.flomen.cnn_files import training_file as flomen_files
from cnn.preprocessing.vgg_preprocessing import preprocess_image as preprocess_vgg_image
import tensorflow as tf


height, width = 224, 224

def preprocess_image(image_path, create_path):
  
  with tf.Session() as sess:
    
    test_image_file = tf.gfile.FastGFile(image_path, 'rb').read()
    print(tf.shape(test_image_file))
    test_image = tf.image.decode_jpeg(test_image_file, channels=3)
    print(tf.shape(test_image_file))
    test_image = preprocess_vgg_image(test_image, height, width, is_training=True)
    
    print(tf.shape(test_image_file))
    tf.gfile.FastGFile(create_path, 'w').write(test_image.SerializeToString())
    
    print(sess)

if __name__ == '__main__':
  
  cnn_file = flomen_files()
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  create_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  preprocess_image(test_file_path, create_file_path)