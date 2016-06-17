'''
Created on Jun 16, 2016

@author: Levan Tsinadze
'''

import tensorflow as tf
import sys

from cnn_functions import conv_net
from weights_biases import weights
from weights_biases import biases

from parameters_saver import parameters_file_conv as model_path_new
from parameters_saver import parameters_file_conv_saved as model_path_saved
from cnn_image_reader import read_image

IMAGE_SIZE = 28
FILE_PATH = '/storage/ann/digits/'
SAVE_FILE = '/storage/ann/digits/parameters/conv_model.ckpt'


n_input = 784  # MNIST data input (img shape: 28*28)
x = tf.placeholder(tf.float32, [None, n_input])

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

pred = conv_net(x, weights, biases, keep_prob)

# Evaluate model
recognize_image = tf.argmax(pred, 1)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
model_path = ''

if len(sys.argv) > 1 and sys.argv[1] == '1':
    model_path = model_path_new
else:
    model_path = model_path_saved
    
# dig_bytes = open(image_path, "rb").read()
# print mnist.test.images[15]

with tf.Session() as sess:
    print 'Start session'
    # Initialize variables
    load_path = saver.restore(sess, model_path)
    print "Model restored from file: %s" % model_path
    image_rec = read_image()
    # Calculate accuracy for 256 mnist test images
    resp_dgt = sess.run(recognize_image, feed_dict={x: image_rec,
                                      keep_prob: 0.75})
    print "Recognized image:", resp_dgt[0]
