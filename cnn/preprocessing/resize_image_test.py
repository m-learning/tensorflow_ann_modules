'''
Created on Oct 17, 2016
Test for image resizing
@author: Levan Tsinadze
'''

from cnn.flomen.cnn_files import training_file as flomen_files
from cnn.preprocessing.vgg_preprocessing import preprocess_image as preprocess_vgg_image
import cv2
import tensorflow as tf


try:
  from PIL import Image
except ImportError:
  print "Importing Image from PIL threw exception"
  import Image

height, width = 224, 224
vgg_size = (224, 224)

def preprocess_image(image_path, create_path):
  
  with tf.Session() as sess:
    
    im = cv2.imread(image_path, 1)
    im_h, im_w = im.shape[:2]
    if im_h < 224 or im_w < 244:
      resized_image = cv2.resize(im, vgg_size, interpolation=cv2.INTER_CUBIC)
    elif im_h > 244 and im_w > 244:
      resized_image = cv2.resize(im, vgg_size, interpolation=cv2.INTER_AREA)
    else:
      resized_image = im
    cv2.imshow('img', resized_image)
    cv2.imwrite(create_path, resized_image)
    test_image_file = tf.gfile.FastGFile(create_path, 'rb').read()
    print tf.shape(test_image_file)
    test_image = tf.image.decode_jpeg(test_image_file, channels=3)
    print tf.shape(test_image_file)
    test_image = preprocess_vgg_image(test_image, height, width, is_training=True)
    
    print tf.shape(test_image_file)
    
    print sess

if __name__ == '__main__':
  
  cnn_file = flomen_files()
  test_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'test_image.jpg')
  create_file_path = cnn_file.join_path(cnn_file.get_or_init_test_dir(), 'resized_image.jpg')
  preprocess_image(test_file_path, create_file_path)
