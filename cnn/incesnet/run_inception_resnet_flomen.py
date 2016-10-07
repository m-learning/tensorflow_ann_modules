# '''
# Created on Oct 3, 2016
# Runs inception-ResNet-v2 module with checkpoint
# @author: levan-lev
# '''

from __future__ import absolute_import
from __future__ import division

import sys

from cnn.flomen.cnn_files import training_file as flomen_files
from cnn.flowers.cnn_files import training_file as flower_files
import cnn.incesnet.inception_resnet_v2 as inception_resnet_v2
from cnn.incesnet.run_inception_resnet_general import inception_resnet_v2_general_interface
from cnn.preprocessing.inception_preprocessing import preprocess_for_eval
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = 299, 299

# Runs Inception-ResNet-v2 Module
class inception_resnet_v2_interface(inception_resnet_v2_general_interface):
  
  def __init__(self, cnn_file):
    super(inception_resnet_v2_interface, self).__init__(cnn_file)
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
          inputs = tf.random_uniform((batch_size, height, width, 3))
          end_interface = inception_resnet_v2.inception_resnet_v2_interface(inputs, num_classes=8,
                                                                            is_training=False)
          
          init_fn = slim.assign_from_checkpoint_fn(self.checkpoint_dir,
                                                   slim.get_model_variables('InceptionResnetV2'))
          with tf.Session() as sess:
              
              init_fn(sess)
      
              test_image_file = tf.gfile.FastGFile(image_path, 'rb').read()
              test_image = tf.image.decode_jpeg(test_image_file, channels=3)
              test_image = preprocess_for_eval(test_image, height, width)
      
              _, predictions = sess.run([test_image, end_interface])
              self.print_answer(predictions)
              
if __name__ == '__main__':
  
  if len(sys.argv) > 1 and sys.argv[1] == 'flowers' :
    cnn_file = flower_files()
  else:
    cnn_file = flomen_files()
  resnet_interface = inception_resnet_v2_interface(cnn_file)
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  resnet_interface.run_interface(test_file_path)
