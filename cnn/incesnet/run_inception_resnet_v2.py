# '''
# Created on Oct 3, 2016
# Runs inception-ResNet-v2 module with checkpoint
# @author: levan-lev
# '''

import sys

from cnn.flomen.cnn_files import training_file as flomen_files
from cnn.flowers.cnn_files import training_file as flower_files
import cnn.incesnet.inception_resnet_v2 as inception_resnet_v2
import tensorflow as tf


slim = tf.contrib.slim

batch_size = 3
image_size = 299

# Runs Inception-ResNet-v2 Module
class inception_resnet_v2_interface(object):
  
  def __init__(self, cnn_file, checkpoint_file):
    self.checkpoint_dir = cnn_file.join_path(cnn_file.init_files_directory(), checkpoint_file)
  
  # Runs recognition on passed image path
  def run_interface(self, image_path):
    
    with tf.Graph().as_default():

      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
          logits, _ = inception_resnet_v2.inception_resnet_v2([1, 299, 299, 3], num_classes=8, is_training=False)
          probabilities = tf.nn.softmax(logits)
      
          init_fn = slim.assign_from_checkpoint_fn(self.checkpoint_dir,
          slim.get_model_variables('InceptionResnetV2'))
      
          with tf.Session() as sess:
              
              init_fn(sess)
      
              test_image_string = tf.gfile.FastGFile(image_path, 'rb').read()
              test_image = tf.image.decode_jpeg(test_image_string, channels=3)
      
              _, probabilities = sess.run([test_image, probabilities])
              print probabilities
              probabilities = probabilities[0, 0:]
              sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
      
              print sorted_inds
              # names = flowers.create_readable_names_for_imagenet_labels()
              # for i in range(8):
              #    index = sorted_inds[i]
              #    print((probabilities[index], names[index]))
              
if __name__ == '__main__':
  
  if len(sys.argv) > 1 and sys.argv[1] == 'flowers' :
    cnn_file = flower_files()
  else:
    cnn_file = flomen_files()
  resnet_interface = inception_resnet_v2_interface(cnn_file, 'model.ckpt-2000')
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  resnet_interface.run_interface(test_file_path)
