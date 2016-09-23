'''
Created on Sep 23, 2016
General configuration parameters for training testing and evaluation
@author: Levan Tsinadze
'''

import cnn.incesnet.training_parameters as FLAGS
import cnn.incesnet.evaluation_parameters as EVAL_FLAGS

# Prepares flowers for inception
def define_training_parameters(file_mngr, dataset_name):
  
  file_mngr.get_or_init_training_set()
  FLAGS.train_dir = file_mngr.init_files_directory()
  FLAGS.dataset_name = dataset_name
  FLAGS.dataset_split_name = 'train'
  FLAGS.dataset_dir = file_mngr.get_data_directory()
  FLAGS.checkpoint_path = file_mngr.join_path(file_mngr.init_files_directory, 'inception_resnet_v2')
  FLAGS.checkpoint_exclude_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits/Logits'
  FLAGS.trainable_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits/Logits'
  
  FLAGS.max_number_of_steps = 1000
  FLAGS.batch_size = 32
  FLAGS.learning_rate = 0.01
  FLAGS.learning_rate_decay_type = 'fixed'
  FLAGS.save_interval_secs = 60
  FLAGS.save_summaries_secs = 60
  FLAGS.log_every_n_steps = 100
  FLAGS.optimizer = 'rmsprop'
  FLAGS.weight_decay = 0.00004
  

# Prepares evaluation parameters
def define_eval_parameters(file_mngr, dataset_name):

  EVAL_FLAGS.checkpoint_path = file_mngr.init_files_directory()
  EVAL_FLAGS.eval_dir = file_mngr.init_files_directory()
  EVAL_FLAGS.dataset_name = dataset_name
  EVAL_FLAGS.dataset_split_name = 'validation'
  EVAL_FLAGS.dataset_dir = file_mngr.get_data_directory()
  EVAL_FLAGS.model_name = 'inception_resnet_v2'