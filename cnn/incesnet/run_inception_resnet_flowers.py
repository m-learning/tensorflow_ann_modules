# '''
# Created on Oct 4, 2016
# Runs inception-ResNet-v2 module with checkpoint
# @author: levan-lev
# '''

from __future__ import absolute_import
from __future__ import division

from cnn.incesnet.run_inception_resnet_general import inception_resnet_v2_general_interface
from cnn.flowers.cnn_files import training_file as flower_files
import cnn.incesnet.inception_resnet_v2 as inception_resnet_v2
from cnn.preprocessing.inception_preprocessing import preprocess_for_eval
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = 299, 299

# Runs Inception-ResNet-v2 Module
class inception_resnet_flowers_interface(inception_resnet_v2_general_interface):
  
  def __init__(self, cnn_file, checkpoint_file):
    super(inception_resnet_flowers_interface, self).__init__(cnn_file, checkpoint_file)
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
          inputs = tf.random_uniform((batch_size, height, width, 3))
          _, endpoints = inception_resnet_v2.inception_resnet_v2(inputs, num_classes=5, is_training=False)
          end_interface = endpoints[inception_resnet_v2.END_POINT_KEY]
      
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
  
  cnn_file = flower_files()
  resnet_interface = inception_resnet_flowers_interface(cnn_file, 'model.ckpt-1002')
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  resnet_interface.run_interface(test_file_path)
