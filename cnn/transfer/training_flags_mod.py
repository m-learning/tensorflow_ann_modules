'''
Created on Jun 28, 2016

Initializes training flags

@author: Levan Tsinadze
'''

# Defines training process directories
IMAGENET_DIR = 'imagenet'
BOTTLENECK_DIR = 'bottleneck'

# Details of the training configuration.
how_many_training_steps = 10000  # How many training steps to run before ending
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
summaries_dir = 'retrain_inception_logs'  # Where to save summary logs 
          # for TensorBoard

# Training and testing flags
class training_flags(object):
  """
    Class to define and configure training flags and hyper parameters
  """
  
  # Initializes flags for training
  def __init__(self, tr_files):

    # Training data and cache directories
    prnt_dir = tr_files.get_data_general_directory()
    
    # Input and output file flags.
    self.image_dir = tr_files.get_data_directory()  # Path to folders of labeled images
    self.output_graph = tr_files.get_or_init_files_path()  # Where to save the trained graph
    self.output_labels = tr_files.get_or_init_labels_path()  # Where to save the trained graph's labels
    
    # File-system cache locations.
    self.model_dir = tr_files.join_path(prnt_dir , IMAGENET_DIR)  # Path to classify_image_graph_def.pb, """
                                    # imagenet_synset_to_human_label_map.txt, and
                                    # imagenet_2012_challenge_label_map_proto.pbtxt
    self.bottleneck_dir = tr_files.join_path(prnt_dir , BOTTLENECK_DIR)  # Path to cache bottleneck layer values as files
    
# Generates and initializes flags for training and testing
def init_flaged_data(tr_files):
  return training_flags(tr_files)
