"""
Created on Jun 28, 2016

Initializes training flags

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from cnn.utils import cnn_flags_utils as conts
from cnn.utils import file_utils


# Defines training process directories
IMAGENET_DIR = 'imagenet'
BOTTLENECK_DIR = 'bottleneck'

# Key for weight decays
LOSSES = 'losses'

# Keep probability for "dropout" layers
keep_prob = 0.5

# L2 regularization weight decay
weight_decay = 0.00004

# Details of the training configuration.
how_many_training_steps = 25000  # How many training steps to run before ending
learning_rate = 0.01  # How large a learning rate to use when training
testing_percentage = 10  # What percentage of images to use as a test set
validation_percentage = 10  # What percentage of images to use as a validation set
eval_step_interval = 10  # How often to evaluate the training results
train_batch_size = 100  # How many images to train on at a time
test_batch_size = 500  # How many images to test on at a time. This
                            # test set is only used infrequently to verify
# the overall accuracy of the model.
validation_batch_size = 100
    # How many images to use in an evaluation batch. This validation set is
    # used much more often than the test set, and is an early indicator of
    # how accurate the model is during training

# Controls the distortions used during training.
flip_left_right = False

# Whether to randomly flip half of the training images horizontallyS
random_crop = 0
    # A percentage determining how much of a margin to randomly crop off the
# training images
random_scale = 0
    # A percentage determining how much to randomly scale up the size of the
# training images by
random_brightness = 0
    # A percentage determining how much to randomly multiply the training
# image input pixels up or down by
final_tensor_name = 'final_result'  # The name of the output classification layer in
    # the retrained graph

log_board_data = True    

summaries_dir = 'retrain_inception_logs'  # Where to save summary logs 
          # for TensorBoard

########################################
# Parameters for runtime configuration #
########################################

# Training data and cache directories
prnt_dir = None

# Input and output file flags.
image_dir = None  # Path to folders of labeled images
output_graph = None  # Where to save the trained graph
output_labels = None  # Where to save the trained graph's labels
print_dataset = None  # Flag to log dataset files array
# File-system cache locations.
model_dir = None  # Path to classify_image_graph_def.pb, """
                                # imagenet_synset_to_human_label_map.txt, and
                                # imagenet_2012_challenge_label_map_proto.pbtxt
bottleneck_dir = None  # Path to cache bottleneck layer values as files


is_training = True

###########################
# Encapsulated parameters #
###########################

_training_flags_to_set = True

def _set_training_flags(tr_files):
  """Initializes flags for training
    Args:
      tr_files - training files manager
  """
  global _training_flags_to_set
  if _training_flags_to_set is None or _training_flags_to_set:
    global prnt_dir, model_dir
    # Training data and cache directories
    prnt_dir = tr_files.get_data_general_directory()
    # Input and output file flags.
    model_dir = tr_files.join_path(prnt_dir, IMAGENET_DIR)  # Path to classify_image_graph_def.pb
    _training_flags_to_set = False

def retrieve_args(argument_flags, _files):
  """Adds configuration from command line arguments
    Args:
     arg_parser - runtime parameters parser
     _files - files manager
  """
  
  if argument_flags.training_steps:
    global how_many_training_steps
    how_many_training_steps = argument_flags.training_steps
    print('Number of training steps was set as - ' , str(how_many_training_steps))
  
  if argument_flags.keep_prob:
    global keep_prob
    if argument_flags.keep_prob > 1:
      keep_prob = (argument_flags.keep_prob / conts.FACTOR_FOR_KEEP_PROB)
    else:
      keep_prob = argument_flags.keep_prob
    print('Dropout keep probability was set as - ', keep_prob)
  
  if argument_flags.learning_rate:
    global learning_rate
    learning_rate = argument_flags.learning_rate
    print('Learning rate was set as - ', learning_rate)
  
  global image_dir, output_graph, output_labels, bottleneck_dir
  if argument_flags.image_dir:
    image_dir = argument_flags.image_dir
    file_utils.ensure_dir_exists(image_dir)
    print('Image directory was set as - ' , image_dir)
  else:
    image_dir = _files.get_data_directory()  # Path to folders of labeled images
    
  if argument_flags.output_graph:
    file_utils.ensure_dir_exists(argument_flags.output_graph)
    output_graph = _files.join_path(argument_flags.output_graph,
                                      file_utils.WEIGHTS_FILE)
    output_labels = _files.join_path(argument_flags.output_graph,
                                       file_utils.LABELS_FILE)
    print('Output graph path was set as - ' , output_graph)
    print('Output labels path was set as - ' , output_labels)
  else:
    output_graph = _files.get_or_init_files_path()  # Where to save the trained graph
    output_labels = _files.get_or_init_labels_path()  # Where to save the trained graph's labels
  
  # File-system cache locations.
  global _training_flags_to_set
  _training_flags_to_set = True
  _set_training_flags(_files)
  if argument_flags.bottleneck_dir:
    bottleneck_dir = argument_flags.bottleneck_dir
    print('Bottleneck path was set as - ' , bottleneck_dir)
  else:
    bottleneck_dir = _files.join_path(prnt_dir , BOTTLENECK_DIR)
  
  if argument_flags.print_dataset:
    global print_dataset
    print_dataset = True
  
def parse_eval_arguments():
  """Parses command line arguments for evaluation
    Returns:
      argument_flags - command line arguments
  """
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--image_path',
                          type=str,
                          help='Image path for recognition')
  (argument_flags, _) = arg_parser.parse_known_args()
  
  return argument_flags
                          
def parse_and_retrieve(tr_files=None):
  """Retrieves command line arguments
    Args:
      tr_files - File utility object for training data preparation
  """
  
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--training_steps',
                          type=int,
                          help='Number of training iterations')
  arg_parser.add_argument('--keep_prob',
                          type=float,
                          help='Dropout keep probability')
  arg_parser.add_argument('--learning_rate',
                          type=float,
                          help='Learning rate') 
  arg_parser.add_argument('--image_dir',
                          type=str,
                          help='Path to folders of labeled images.')
  arg_parser.add_argument('--output_graph',
                          type=str,
                          help='Where to save the trained graph.')
  arg_parser.add_argument('--bottleneck_dir',
                          type=str,
                          help='Path to cache bottleneck layer values as files.')
  arg_parser.add_argument('--print_dataset',
                          dest='print_dataset',
                          action='store_true',
                          help='Prints data set file names and labels.')
  arg_parser.add_argument('--not_print_dataset',
                          dest='print_dataset',
                          action='store_false',
                          help='Do not print data set file names and labels.')
  (argument_flags, _) = arg_parser.parse_known_args()
  retrieve_args(argument_flags, tr_files)

def init_flaged_data(tr_files):
  """Generates and initializes flags for 
    training and testing
    Args:
      tr_files - Training files management utility
    Returns:
      training_flags configured instance
  """
  _set_training_flags(tr_files)
