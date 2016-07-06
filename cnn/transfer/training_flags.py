'''
Created on Jun 28, 2016

Initializes training flags

@author: Levan Tsinadze
'''

IMAGENET_DIR = 'imagenet'
BOTTLENECK_DIR = 'bottleneck'

# Training and testing flags
class training_flags:
  
  def __init__(self, tr_files):

    # Training data and cache directories
    self.prnt_dir = tr_files.get_data_general_directory()
    self.tr_dir = tr_files.get_data_directory()
    self.out_grph = tr_files.get_or_init_files_path()
    self.out_lbl = tr_files.get_or_init_labels_path()
    self.mdl_dir = tr_files.join_path(self.prnt_dir , IMAGENET_DIR)
    self.btl_dir = tr_files.join_path(self.prnt_dir , BOTTLENECK_DIR)
    

  # Initializes flags for training
  def init_flags(self):
    
    # Input and output file flags.
    self.image_dir = self.tr_dir  # Path to folders of labeled images
    self.output_graph = self.out_grph  # Where to save the trained graph
    self.output_labels = self.out_lbl  # Where to save the trained graph's labels
    
    # Details of the training configuration.
    self.how_many_training_steps = 4000  # How many training steps to run before ending
    self.learning_rate = 0.01  # How large a learning rate to use when training
    self.testing_percentage = 10  # What percentage of images to use as a test set
    self.validation_percentage = 10  # What percentage of images to use as a validation set
    self.eval_step_interval = 10  # How often to evaluate the training results
    self.train_batch_size = 100  # How many images to train on at a time
    self.test_batch_size = 500  # How many images to test on at a time. This
                                # test set is only used infrequently to verify
                                # the overall accuracy of the model.
    self.validation_batch_size = 100
        # How many images to use in an evaluation batch. This validation set is"""
        # used much more often than the test set, and is an early indicator of
        # how accurate the model is during training
    
    # File-system cache locations.
    self.model_dir = self.mdl_dir  # Path to classify_image_graph_def.pb, """
                                    # imagenet_synset_to_human_label_map.txt, and
                                    # imagenet_2012_challenge_label_map_proto.pbtxt
    self.bottleneck_dir = self.btl_dir  # Path to cache bottleneck layer values as files
    self.final_tensor_name = 'final_result'
                                    # The name of the output classification layer in
                                    # the retrained graph
    
    # Controls the distortions used during training.
    self.flip_left_right = False
        # Whether to randomly flip half of the training images horizontallyS
    self.random_crop = 0
        # A percentage determining how much of a margin to randomly crop off the"""
        # training images
    self.random_scale = 0
        # A percentage determining how much to randomly scale up the size of the
        # training images by
    self.random_brightness = 0
        # A percentage determining how much to randomly multiply the training"""
        # image input pixels up or down by.""")

# Generates and initializes flags for training and testing
def init_flaged_data(tr_files):
  
  trn_flags = training_flags(tr_files)
  trn_flags.init_flags()
  
  return trn_flags
  
        
    
