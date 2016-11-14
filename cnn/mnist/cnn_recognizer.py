"""
Created on Jun 25, 2016
Recognizes for image
@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn.utils.cnn_flags_utils import KEEP_FULL_PROB
from cnn.mnist.cnn_files import training_file
from cnn.mnist.cnn_input_reader import read_input_file
from cnn.mnist.cnn_network import cnn_functions
import tensorflow as tf


class image_recognizer(object):
  """Class to generate and run recognizer interface"""
  
  def init_feed_parameter(self, image_rec):
    """Initializes parameters for recognition interface
      Args:
        image_rec - image to recognize
      Returns:
        rec_feed - parameters for recognition interface
    """
    rec_feed = {self.x: image_rec, self.keep_prob: KEEP_FULL_PROB}
    return rec_feed
    
  def recognize_image(self, image_file_path):
    """Recognizes digit from file
      Args:
        image_file_path - path for image file
      Return:
        Recognized digit
    """
  
    network = cnn_functions()
    pred = network.conv2d
    
    # Evaluate model
    recognize_image = tf.argmax(pred, 1)
    # Initializing saver to read trained data
    saver = tf.train.Saver()
    tf.initialize_all_variables()
    tr_files = training_file()
    model_path = tr_files.get_or_init_files_path()
    with tf.Session() as sess:
        print('Start session')
        # Initialize variables
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        image_rec = read_input_file(image_file_path)
        rec_feed = self.init_feed_parameter(image_rec)
        # Recognize image
        resp_dgt = sess.run(recognize_image, feed_dict=rec_feed)
        print("Recognized image:", resp_dgt[0])
    
    return resp_dgt[0]
    
    
