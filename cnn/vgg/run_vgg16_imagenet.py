# '''
# Created on Oct 4, 2016
# Runs VGG module with checkpoint
# @author: levan-lev
# '''

from __future__ import absolute_import
from __future__ import division

from PIL import Image

from cnn.datasets import imagenet
from cnn.flomen.cnn_files import training_file as flomen_files
import cnn.nets.run_network as general_network
from cnn.preprocessing.vgg_preprocessing import preprocess_image
import cnn.vgg.vgg as vgg
import cv2
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = 224, 224
network_interface = vgg.vgg_16_fc

# Runs VGG Module
class vgg_interface(object):
  
  def __init__(self, cnn_file, checkpoint_file):
    self.checkpoint_dir = cnn_file.join_path(cnn_file.init_files_directory(), checkpoint_file)
    
  # Prints predicted answer
  def print_answer(self, predict_values):
    
    predictions = np.squeeze(predict_values)
    top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions

    names = imagenet.create_readable_names_for_imagenet_labels()
    print top_k
    for node_id in top_k:
      print node_id
      print((str(node_id), predictions[node_id], names[node_id]))
  
  # Run model in an other way
  def run_interface_other(self, image_path):
    
    sample_images = [image_path]
    # Load the model
    sess = tf.Session()
    arg_scope = vgg.vgg_arg_scope()
    with slim.arg_scope(arg_scope):
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, end_points = network_interface(inputs, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, self.checkpoint_dir)
    end_interface = end_points[general_network.layer_key]
    for image in sample_images:
      im = Image.open(image).resize((height, width))
      im = np.array(im)
      im = im.reshape(-1, height, width, 3)
      im = 2 * (im / 255.0) - 1.0
      # image_to_test = preprocess_for_eval(im, height, width)
      predict_values, _ = sess.run([end_interface, logits], feed_dict={inputs: im})
      self.print_answer(predict_values)
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(vgg.vgg_arg_scope()):
          inputs = tf.random_uniform((batch_size, height, width, 3))
          end_interface = general_network.interface_function(inputs,
                                                             num_classes=1000,
                                                             is_training=False)
          print image_path
          print self.checkpoint_dir
          
          print end_interface
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
      logits, end_points = general_network.interface_function(scaled_input_tensor, is_training=False,
                                                              spatial_squeeze=True)
    saver = tf.train.Saver()
    saver.restore(sess, self.checkpoint_dir)
    
    for image in sample_images:
      im = cv2.resize(cv2.imread(image), (height, width)).astype(np.float32)
      im[:, :, 0] -= 103.939
      im[:, :, 1] -= 116.779
      im[:, :, 2] -= 123.68
      im = im.transpose((2, 0, 1))
      im = np.expand_dims(im, axis=0)
      predict_values, logit_values = sess.run([end_points[general_network.layer_key], logits],
                                              feed_dict={input_tensor: im})
      # soft_max_values = tf.nn.softmax(logit_values, 'Endpoint')
      print (np.max(predict_values), np.max(logit_values))
      print (np.argmax(predict_values), np.argmax(logit_values))
      self.print_answer(predict_values)
              
if __name__ == '__main__':
  
  cnn_file = flomen_files()
  app_interface = vgg_interface(cnn_file, 'vgg_16.ckpt')
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  app_interface.run_interface(test_file_path)
