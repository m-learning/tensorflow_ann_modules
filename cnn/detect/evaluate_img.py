'''
Created on Aug 8, 2016

Evaluates object detection in image

@author: Levan Tsinadze
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import subprocess
from scipy.misc import imread
from cnn.detect import cnn_files
%matplotlib inline

from train import build_forward
from cnn.detect.utils import googlenet_load, train_utils
from cnn.detect.utils.annolist import AnnotationLib as al
from cnn.detect.utils.stitch_wrapper import stitch_rects
from cnn.detect.utils.train_utils import add_rectangles
from cnn.detect.utils.rect import Rect
from cnn.detect.utils.stitch_wrapper import stitch_rects
from evaluate import add_rectangles
import cv2

import cnn.detect.cnn_files

hypes_file = './hypes/overfeat_rezoom.json'
iteration = 150000
with open(hypes_file, 'r') as f:
    H = json.load(f)
true_idl = './data/brainwash/brainwash_val.idl'
pred_idl = './output/%d_val_%s.idl' % (iteration, os.path.basename(hypes_file).replace('.json', ''))
true_annos = al.parse(true_idl)

#Loads graph
tf.reset_default_graph()
googlenet = googlenet_load.init(H)
x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
if H['use_rezoom']:
    pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    grid_area = H['grid_height'] * H['grid_width']
    pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
    if H['reregress']:
        pred_boxes = pred_boxes + pred_boxes_deltas
else:
    pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess_files = cnn_files()
    saver.restore(sess, sess_files.get_checkpoint(iteration))

    annolist = al.AnnoList()
    import time; t = time.time()
    for i in range(0, 500):
        true_anno = true_annos[i]
        img = imread('./data/brainwash/%s' % true_anno.imageName)
        feed = {x_in: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
        pred_anno = al.Annotation()
        pred_anno.imageName = true_anno.imageName
        new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.3)
    
        pred_anno.rects = rects
        annolist.append(pred_anno)

        if i % 10 == 0 and i < 200:
            pass
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(new_img)
        if i % 100 == 0:
            print(i)
    avg_time = (time.time() - t) / (i + 1)
    print('%f images/sec' % (1. / avg_time))