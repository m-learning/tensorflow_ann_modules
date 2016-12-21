'''
Created on Aug 8, 2016

Evaluates object detection in image

@author: Levan Tsinadze
'''

import json
import sys

from scipy.misc import imread, imresize

from cnn.detect.cnn_files import training_file
from cnn.detect.utils import googlenet_load
from cnn.detect.utils.train_utils import add_rectangles
import matplotlib.pyplot as plt
import tensorflow as tf
from train import build_forward


def generate_pred_image(pred_params):
    
    (H, img_path, pred_boxes, _, pred_confidences) = pred_params
    img = imread(img_path)
    print img
    res_img = imresize(img, (H["image_height"], H["image_width"]), interp='cubic')
    feed = {x_in: res_img}
    (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
    new_img, _ = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                    use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.3)

    _ = plt.figure(figsize=(12, 12))
    plt.imshow(new_img)
    print new_img
    
# Initializes image path
def init_image_path(sys_params, cnn_param_files):
  
  if len(sys_params) >= 2:
    test_image_path = cnn_param_files.join_path(cnn_param_files.cnn_param_files, sys_params[1])
  else:
    test_image_path = cnn_param_files.get_or_init_test_path()
    
  return test_image_path
  

cnn_param_files = training_file()
hypes_file = cnn_param_files.get_hypes_file()
iteration = 190000
with open(hypes_file, 'r') as f:
    H = json.load(f)

# Loads graph
tf.reset_default_graph()
googlenet = googlenet_load.init(H)
x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
if H['use_rezoom']:
    (pred_boxes, pred_logits, pred_confidences,
     pred_confs_deltas, pred_boxes_deltas) = build_forward(H, tf.expand_dims(x_in, 0),
                                                           googlenet, 'test', reuse=None)
    grid_area = H['grid_height'] * H['grid_width']
    pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas,
                                                           [grid_area * H['rnn_len'], 2])),
                                                           [grid_area, H['rnn_len'], 2])
    if H['reregress']:
        pred_boxes = pred_boxes + pred_boxes_deltas
else:
    pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0),
                                                              googlenet, 'test', reuse=None)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, cnn_param_files.get_checkpoint(iteration))
    img_path = init_image_path(sys.argv, cnn_param_files)
    pred_params = (H, img_path, pred_boxes, pred_logits, pred_confidences)
    generate_pred_image(pred_params)
    print 'Image processing is finnished'
