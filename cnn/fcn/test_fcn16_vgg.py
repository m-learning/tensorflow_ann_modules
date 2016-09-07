#!/usr/bin/env python

import skimage.io

import os
import scipy as scp

import logging
import tensorflow as tf
import sys

import cnn.fcn.fcn16_vgg as fcn16_vgg
import cnn.fcn.utils as utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

test_dir = utils.get_test_dir()
test_result_dir = utils.get_test_results_dir()
img1 = skimage.io.imread(os.path.join(utils.get_test_dir(), 'tabby_cat.png'))

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn16_vgg.FCN16VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    fcn16_downsampled = os.path.join(test_result_dir, 'fcn16_downsampled.png')
    fcn16_upsampled = os.path.join(test_result_dir, 'fcn16_upsampled.png')
    scp.misc.imsave(fcn16_downsampled, down_color)
    scp.misc.imsave(fcn16_upsampled, up_color)
