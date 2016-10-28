"""
Created on Jun 28, 2016

Initializes training flags

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Defines training process directories
IMAGENET_DIR = 'imagenet'
BOTTLENECK_DIR = 'bottleneck'

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

# File-system cache locations.
model_dir = None  # Path to classify_image_graph_def.pb, """
                                # imagenet_synset_to_human_label_map.txt, and
                                # imagenet_2012_challenge_label_map_proto.pbtxt
bottleneck_dir = None  # Path to cache bottleneck layer values as files

def _set_training_flags(tr_files):
  """Initializes flags for training
    Args:
      tr_files - training files utility
  """

  global prnt_dir, image_dir, output_graph, \
         output_labels, model_dir, bottleneck_dir
  # Training data and cache directories
  prnt_dir = tr_files.get_data_general_directory()
  # Input and output file flags.
  image_dir = tr_files.get_data_directory()  # Path to folders of labeled images
  output_graph = tr_files.get_or_init_files_path()  # Where to save the trained graph
  output_labels = tr_files.get_or_init_labels_path()  # Where to save the trained graph's labels
  # File-system cache locations.
  model_dir = tr_files.join_path(prnt_dir, IMAGENET_DIR)  # Path to classify_image_graph_def.pb, """
                                  # imagenet_synset_to_human_label_map.txt, and
                                  # imagenet_2012_challenge_label_map_proto.pbtxt
  bottleneck_dir = tr_files.join_path(prnt_dir , BOTTLENECK_DIR)  # Path to cache bottleneck layer values as files

def retrieve_args(sys_argv):
  """Adds configuration from system arguments
    Args:
     sys_argv - runtime parameters
  """
  
  if len(sys_argv) > 1:
    global how_many_training_steps
    how_many_training_steps = int(sys_argv[1])
    print('Number of raining steps is set as - ' , str(how_many_training_steps))

def init_flaged_data(tr_files):
  """Generates and initializes flags for 
    training and testing
    Args:
      tr_files - trainign files management utility
    Returns:
      training_flags configured instance
  """
  _set_training_flags(tr_files)
