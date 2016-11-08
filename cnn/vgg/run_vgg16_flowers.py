# '''
# Created on Oct 4, 2016
# Runs VGG module with checkpoint
# @author: levan-lev
# '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

from cnn.flowers.cnn_files import training_file as flower_files
from cnn.nets.run_network import network_interface
import cnn.nets.run_network as general_network
from cnn.preprocessing.vgg_preprocessing import preprocess_image
from cnn.vgg import vgg_resizer
import cnn.vgg.vgg as vgg
from cnn.vgg.vgg_resizer import vgg_image_resizer
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = vgg_resizer.vgg_dim, vgg_resizer.vgg_dim

# Runs Inception-ResNet-v2 Module
class vgg_interface(network_interface):
  
  def __init__(self, cnn_file):
    super(vgg_interface, self).__init__(cnn_file)
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(vgg.vgg_arg_scope()):
          inputs = tf.random_uniform((batch_size, height, width, 3))
          end_interface = general_network.interface_function(inputs, num_classes=5, is_training=False)
          init_fn = slim.assign_from_checkpoint_fn(self.checkpoint_dir,
                                                   slim.get_model_variables(general_network.network_name))
          with tf.Session() as sess:
              
              init_fn(sess)
      
              test_image_file = tf.gfile.FastGFile(image_path, 'rb').read()
              test_image = tf.image.decode_jpeg(test_image_file, channels=3)
              test_image = preprocess_image(test_image, height, width)
      
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
    arg_scope = vgg.vgg_arg_scope()
    with slim.arg_scope(arg_scope):
      logits, end_points = general_network.interface_function(scaled_input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, self.checkpoint_dir)
    
    for image in sample_images:
      im = Image.open(image).resize((height, width))
      im = np.array(im)
      im = im.reshape(-1, height, width, 3)
      predict_values, logit_values = sess.run([end_points[general_network.layer_key], logits],
                                              feed_dict={input_tensor: im})
      print(np.max(predict_values), np.max(logit_values))
      print(np.argmax(predict_values), np.argmax(logit_values))
      self.print_answer(predict_values)
              
if __name__ == '__main__':
  
  cnn_file = flower_files(vgg_image_resizer())
  app_interface = vgg_interface(cnn_file)
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  app_interface.run_interface(test_file_path)
