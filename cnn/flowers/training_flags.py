'''
Created on Jun 28, 2016

Initializes training flags

@author: Levan Tsinadze
'''

import tensorflow as tf

import cnn.flowers.cnn_files as tr_datas
from cnn.flowers.cnn_files import training_file

def init_flaged_data():
    
  FLAGS = tf.app.flags.FLAGS
    
  tr_files = training_file()
  
  prnt_dir = tr_files.get_current() + tr_datas.PATH_CNN_DIRECTORY
  tr_dir = tr_files.get_data_directory()
  out_grph = tr_files.get_or_init_files_path()
  out_lbl = tr_files.get_or_init_labels_path()
  mdl_dir = prnt_dir + 'imagenet'
  btl_dir = prnt_dir + 'bottleneck'
      # Input and output file flags.
  tf.app.flags.DEFINE_string('image_dir', tr_dir, """Path to folders of labeled images.""")
  tf.app.flags.DEFINE_string('output_graph', out_grph, """Where to save the trained graph.""")
  tf.app.flags.DEFINE_string('output_labels', out_lbl, """Where to save the trained graph's labels.""")
  
  # Details of the training configuration.
  tf.app.flags.DEFINE_integer('how_many_training_steps', 4000, """How many training steps to run before ending.""")
  tf.app.flags.DEFINE_float('learning_rate', 0.01, """How large a learning rate to use when training.""")
  tf.app.flags.DEFINE_integer(
      'testing_percentage', 10, """What percentage of images to use as a test set.""")
  tf.app.flags.DEFINE_integer(
      'validation_percentage', 10, """What percentage of images to use as a validation set.""")
  tf.app.flags.DEFINE_integer('eval_step_interval', 10, """How often to evaluate the training results.""")
  tf.app.flags.DEFINE_integer('train_batch_size', 100, """How many images to train on at a time.""")
  tf.app.flags.DEFINE_integer('test_batch_size', 500, """How many images to test on at a time. This"""
                              """ test set is only used infrequently to verify"""
                              """ the overall accuracy of the model.""")
  tf.app.flags.DEFINE_integer('validation_batch_size', 100,
      """How many images to use in an evaluation batch. This validation set is"""
      """ used much more often than the test set, and is an early indicator of"""
      """ how accurate the model is during training.""")
  
  # File-system cache locations.
  tf.app.flags.DEFINE_string('model_dir', mdl_dir, """Path to classify_image_graph_def.pb, """
                             """imagenet_synset_to_human_label_map.txt, and """
                             """imagenet_2012_challenge_label_map_proto.pbtxt.""")
  tf.app.flags.DEFINE_string('bottleneck_dir', btl_dir, """Path to cache bottleneck layer values as files.""")
  tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                             """The name of the output classification layer in"""
                             """ the retrained graph.""")
  
  # Controls the distortions used during training.
  tf.app.flags.DEFINE_boolean(
      'flip_left_right', False,
      """Whether to randomly flip half of the training images horizontally.""")
  tf.app.flags.DEFINE_integer(
      'random_crop', 0,
      """A percentage determining how much of a margin to randomly crop off the"""
      """ training images.""")
  tf.app.flags.DEFINE_integer(
      'random_scale', 0,
      """A percentage determining how much to randomly scale up the size of the"""
      """ training images by.""")
  tf.app.flags.DEFINE_integer(
      'random_brightness', 0,
      """A percentage determining how much to randomly multiply the training"""
      """ image input pixels up or down by.""")
  
  return FLAGS