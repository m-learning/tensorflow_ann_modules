#!/usr/bin/env python

import os
import scipy.misc
import skimage
import skimage.io
import skimage.transform
from tensorflow.python.framework import ops

import cnn.fcn.fcn32_vgg as fcn32_vgg
import cnn.fcn.utils as utils
import numpy as np
import scipy as scp
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = ''

test_dir = utils.get_test_dir()
test_result_dir = utils.get_test_results_dir()
img1 = skimage.io.imread(os.path.join(utils.get_test_dir(), 'tabby_cat.png'))

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn32_vgg.FCN32VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)

    print('Finished building Network.')

    init = tf.initialize_all_variables()
    sess.run(tf.initialize_all_variables())

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    down_color = utils.color_image(down[0])
    up_color = utils.color_image(up[0])

    fcn32_downsampled = os.path.join(test_result_dir, 'fcn32_downsampled.png')
    fcn32_upsampled = os.path.join(test_result_dir, 'fcn32_upsampled.png')
    scp.misc.imsave(fcn32_downsampled, down_color)
    scp.misc.imsave(fcn32_upsampled, up_color)
