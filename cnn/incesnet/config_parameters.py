# '''
# Created on Sep 23, 2016
# General configuration parameters for training testing and evaluation
# @author: Levan Tsinadze
# '''

import cnn.incesnet.evaluate_inception_resnet_v2 as eval_inception
import cnn.incesnet.evaluation_parameters as EVAL_FLAGS
import cnn.incesnet.retrain_inception_resnet_v2 as train_inception
import cnn.incesnet.training_parameters as FLAGS


class train_and_eval_config(object):
  
  def __init__(self, file_mngr, dataset_name, dataset_downloader):
    self.file_mngr = file_mngr
    self.dataset_name = dataset_name
    self.dataset_downloader = dataset_downloader

  # Prepares flowers for inception
  def define_training_parameters(self):
    
    self.file_mngr.get_or_init_training_set()
    FLAGS.train_dir = self.file_mngr.init_files_directory()
    FLAGS.dataset_name = self.dataset_name
    FLAGS.dataset_split_name = 'train'
    FLAGS.dataset_dir = self.file_mngr.get_data_directory()
    FLAGS.checkpoint_path = self.file_mngr.join_path(FLAGS.train_dir, 'inception_resnet_v2.ckpt')
    
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
    self.dataset_downloader.run(FLAGS.dataset_dir)
    
  # Prepares evaluation parameters
  def define_eval_parameters(self):
  
    EVAL_FLAGS.checkpoint_path = self.file_mngr.init_files_directory()
    EVAL_FLAGS.eval_dir = self.file_mngr.init_files_directory()
    EVAL_FLAGS.dataset_name = self.dataset_name
    EVAL_FLAGS.dataset_split_name = 'validation'
    EVAL_FLAGS.dataset_dir = self.file_mngr.get_data_directory()
    EVAL_FLAGS.model_name = 'inception_resnet_v2'
    
  
    # Trains network
  def train_net(self):
    self.define_training_parameters()
    train_inception.train_net()
  
  # Evaluates network
  def eval_net(self):
    self.define_eval_parameters()
    eval_inception.eval_net()
  
  # Runs train or evaluation
  def train_or_eval(self, args):
    
    if args[1] == 'eval':
        self.eval_net()
    else:
        self.train_net()
        
  # Traain or eva;luate parameters
  def train_or_eval_net(self, args):
    
    if len(args) > 1:
      self.train_or_eval()
    else:
      self.train_net()
