# '''
# Created on Oct 4, 2016
# Runs inception-ResNet-v2 module with checkpoint
# @author: levan-lev
# '''

from __future__ import absolute_import
from __future__ import division

from PIL import Image

from cnn.datasets import imagenet
from cnn.flomen.cnn_files import training_file as flomen_files
import cnn.incesnet.inception_resnet_v2 as inception_resnet_v2
from cnn.preprocessing.inception_preprocessing import preprocess_for_eval
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = 299, 299

# Runs Inception-ResNet-v2 Module
class inception_resnet_magenet_interface(object):
  
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
    arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, self.checkpoint_dir)
    end_interface = end_points[inception_resnet_v2.END_POINT_KEY]
    for image in sample_images:
      im = Image.open(image).resize((299, 299))
      im = np.array(im)
      im = im.reshape(-1, 299, 299, 3)
      im = 2 * (im / 255.0) - 1.0
      # image_to_test = preprocess_for_eval(im, height, width)
      predict_values, _ = sess.run([end_interface, logits], feed_dict={inputs: im})
      self.print_answer(predict_values)
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
          inputs = tf.random_uniform((batch_size, height, width, 3))
          end_interface = inception_resnet_v2.inception_resnet_v2_interface(inputs, 
                                                                            num_classes=1001, 
                                                                            is_training=False)
          print image_path
          print self.checkpoint_dir
          
          print end_interface
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
  
  cnn_file = flomen_files()
  resnet_interface = inception_resnet_magenet_interface(cnn_file, 'inception_resnet_v2_2016_08_30.ckpt')
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  resnet_interface.run_interface_other(test_file_path)
