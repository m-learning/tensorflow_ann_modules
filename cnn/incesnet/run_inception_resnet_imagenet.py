# '''
# Created on Oct 4, 2016
# Runs inception-ResNet-v2 module with checkpoint
# @author: levan-lev
# '''

from cnn.flomen.cnn_files import training_file as flomen_files
from cnn.datasets import imagenet
import cnn.incesnet.inception_resnet_v2 as inception_resnet_v2
from cnn.preprocessing.inception_preprocessing import preprocess_for_eval
from PIL import Image
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 1
height, width = 299, 299

# Runs Inception-ResNet-v2 Module
class inception_resnet_magenet_interface(object):
  
  def __init__(self, cnn_file, checkpoint_file):
    self.checkpoint_dir = cnn_file.join_path(cnn_file.init_files_directory(), checkpoint_file)
  
  #Run model in an other way
  def run_interface_other(self, image_path):
    
    sample_images = [image_path]
    #Load the model
    sess = tf.Session()
    arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, self.checkpoint_dir)
    for image in sample_images:
      im = Image.open(image).resize((299,299))
      im = np.array(im)
      im = im.reshape(-1,299,299,3)
      image_to_test = preprocess_for_eval(im, height, width)
      predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={inputs: image_to_test})
      print (np.max(predict_values), np.max(logit_values))
      print (np.argmax(predict_values), np.argmax(logit_values))
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
          inputs = tf.random_uniform((batch_size, height, width, 3))
          logits, _ = inception_resnet_v2.inception_resnet_v2(inputs, num_classes=1001, is_training=False)
          probabilities = tf.nn.softmax(logits)
          print logits
          print probabilities
          print image_path
          print self.checkpoint_dir
      
          init_fn = slim.assign_from_checkpoint_fn(self.checkpoint_dir,
                                                   slim.get_model_variables('InceptionResnetV2'))
      
          with tf.Session() as sess:
              
              init_fn(sess)
      
              test_image_string = tf.gfile.FastGFile(image_path, 'rb').read()
              test_image = tf.image.decode_jpeg(test_image_string, channels=3)
              image_to_test = preprocess_for_eval(test_image, height, width)
              _, probabilities = sess.run([image_to_test, probabilities])
              probabilities = probabilities[0, 0:]
              sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
      
              names = imagenet.create_readable_names_for_imagenet_labels()
              for i in range(15):
                index = sorted_inds[i]
                print((str(index), probabilities[index], names[index]))
              
if __name__ == '__main__':
  
  cnn_file = flomen_files()
  resnet_interface = inception_resnet_magenet_interface(cnn_file, 'inception_resnet_v2_2016_08_30.ckpt')
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  resnet_interface.run_interface(test_file_path)
