# '''
# Created on Sep 23, 2016
# General configuration parameters for training testing and evaluation
# @author: Levan Tsinadze
# '''

from __future__ import absolute_import
from __future__ import division

import os
import sys
import tarfile

import cnn.nets.evaluate_network as eval_network
import cnn.nets.evaluation_parameters as EVAL_FLAGS
import cnn.nets.retrain_network as train_network
import cnn.nets.training_parameters as FLAGS
from six.moves import urllib


# Training parameters and data set
class train_and_eval_config(object):
  
  def __init__(self, training_parameters):
    (file_mngr, dataset_name, dataset_downloader,
     train_function, eval_funcion,
     checkpoint_file, checkpoint_url) = training_parameters
    self.file_mngr = file_mngr
    self.dataset_name = dataset_name
    self.dataset_downloader = dataset_downloader
    self.checkpoint_directory = self.file_mngr.init_files_directory()
    self.checkpoint_url = checkpoint_url
    full_checkpoint_file = checkpoint_file + '.ckpt'
    self.checkpoint_file = self.file_mngr.join_path(self.checkpoint_directory, full_checkpoint_file)
    if train_function is None:
      self.train_function = train_network.train_net
    else:
      self.train_function = train_function
    if eval_funcion is None:
      self.eval_function = eval_network.eval_net
    else:
      self.eval_function = eval_funcion
  
  # Gets checkpoint file
  def download_checkpoint(self):
    
    full_checkpoint_url = 'http://download.tensorflow.org/models/' + self.checkpoint_url + '.tar.gz'
    filename = full_checkpoint_url.split('/')[-1]
    filepath = self.file_mngr.join_path(self.checkpoint_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(full_checkpoint_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(self.checkpoint_directory)
  
  # Sets and trainable and exclusion scopes
  def set_trainable_and_exclude_scopes(self, checkpoint_exclude_scopes, trainable_scopes):
    
    if checkpoint_exclude_scopes is not None:
      FLAGS.checkpoint_exclude_scopes = checkpoint_exclude_scopes
    if trainable_scopes is not None:
      FLAGS.trainable_scopes = trainable_scopes
    
  
  # Define checkpoint
  def _set_checkpoint(self):
    FLAGS.checkpoint_path = self.checkpoint_file
  
  # Define hyper - parameters
  def _set_hyper_parameters(self):
    
    FLAGS.max_number_of_steps = 4000
    FLAGS.batch_size = 64
    FLAGS.learning_rate = 0.01
    FLAGS.learning_rate_decay_type = 'exponential'
    FLAGS.save_interval_secs = 60
    FLAGS.save_summaries_secs = 60
    FLAGS.log_every_n_steps = 100
    FLAGS.optimizer = 'rmsprop'
    FLAGS.weight_decay = 0.00004
    
  # Sets network name
  def set_network_name(self, network_name):
      FLAGS.network_name = network_name
  
  # Sets maximum number of steps
  def set_max_number_of_steps(self, max_number_of_steps):
    FLAGS.max_number_of_steps = max_number_of_steps
  
  # Sets network model name
  def set_model_name(self, model_name):
    FLAGS.model_name = model_name
  
  # Sets optimizer function name
  def set_optimizer(self, optimizer):
    FLAGS.optimizer = optimizer
  
  # Sets learning rate decay type
  def set_learning_rate_decay_type(self, learning_rate_decay_type):
    FLAGS.learning_rate_decay_type = learning_rate_decay_type
  
  # Prepares flowers for inception
  def define_training_parameters(self):
    
    self.file_mngr.get_or_init_training_set()
    if not os.path.exists(self.checkpoint_file):
      self.download_checkpoint()
    FLAGS.train_dir = self.file_mngr.init_files_directory()
    FLAGS.dataset_name = self.dataset_name
    FLAGS.dataset_split_name = 'train'
    FLAGS.dataset_dir = self.file_mngr.get_dataset_dir()
    # Adds hyperparameters
    self._set_checkpoint()
    self._set_hyper_parameters()
    # Archive directories
    archive_dir = self.file_mngr.get_archives_directory()
    self.dataset_downloader.run(FLAGS.dataset_dir, archive_dir)
    
  # Prepares evaluation parameters
  def define_eval_parameters(self):
  
    EVAL_FLAGS.checkpoint_path = self.file_mngr.init_files_directory()
    EVAL_FLAGS.eval_dir = self.file_mngr.get_or_init_eval_path()
    EVAL_FLAGS.dataset_name = self.dataset_name
    EVAL_FLAGS.dataset_split_name = 'validation'
    EVAL_FLAGS.dataset_dir = self.file_mngr.get_dataset_dir()
    EVAL_FLAGS.model_name = 'vgg_16'
    
  # Gets configuration function for training and evaluation
  def run_config_function(self, args):
    """
      Runs appropriate configuration function
      Args: system argumenst to decide between 
            training and evaluation configuration
    """
    
    if len(args) > 1 and args[1] == 'eval':
      self.define_eval_parameters()
    else:
      self.define_training_parameters()
  
  # Train or eva;luate parameters
  def train_or_eval_net(self, args):
    
    if len(args) > 1 and len(args) > 1:
      self.eval_function()
    else:
      self.train_function()
