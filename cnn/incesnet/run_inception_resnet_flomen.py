# '''
# Created on Oct 3, 2016
# Runs inception-ResNet-v2 module with checkpoint
# @author: levan-lev
# '''

from __future__ import absolute_import
from __future__ import division

from PIL import Image

from cnn.flomen.cnn_files import training_file as flomen_files
import cnn.incesnet.inception_resnet_v2 as inception_resnet_v2
from cnn.nets.run_network import network_interface
from cnn.preprocessing.inception_preprocessing import preprocess_for_eval
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = 299, 299

# Runs Inception-ResNet-v2 Module
class inception_resnet_v2_interface(network_interface):
  
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
              
  # Runs image classifier
  def run_scaled(self, image_path):
    
    sample_images = [image_path]
    
    input_tensor = tf.placeholder(tf.float32, shape=(None, height, width, 3), name='input_image')
    scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
    scaled_input_tensor = tf.sub(scaled_input_tensor, 0.5)
    scaled_input_tensor = tf.mul(scaled_input_tensor, 2.0)
    
    sess = tf.Session()
    arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
      logits, end_points = inception_resnet_v2.inception_resnet_v2(scaled_input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, self.checkpoint_dir)
    
    for image in sample_images:
      im = Image.open(image).resize((height, width))
      im = np.array(im)
      im = im.reshape(-1, height, width, 3)
      predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
      print (np.max(predict_values), np.max(logit_values))
      print (np.argmax(predict_values), np.argmax(logit_values))
      self.print_answer(predict_values)
              
if __name__ == '__main__':
  
  cnn_file = flomen_files()
  resnet_interface = inception_resnet_v2_interface(cnn_file)
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  resnet_interface.run_interface(test_file_path)
