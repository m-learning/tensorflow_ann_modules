# '''
# Created on Sep 23, 2016
# General configuration parameters for training testing and evaluation
# @author: Levan Tsinadze
# '''

import os
import sys
import tarfile

import cnn.incesnet.evaluate_inception_resnet_v2 as eval_inception
import cnn.incesnet.evaluation_parameters as EVAL_FLAGS
import cnn.incesnet.retrain_inception_resnet_v2 as train_inception
import cnn.incesnet.training_parameters as FLAGS
from six.moves import urllib


# URL to checkpoint file
CHECKPOINT_URL = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
CHECKPOINT_FILE_NAME = 'inception_resnet_v2_2016_08_30.ckpt'

# Training parameters and data set
class train_and_eval_config(object):
  
  def __init__(self, file_mngr, dataset_name, dataset_downloader):
    self.file_mngr = file_mngr
    self.dataset_name = dataset_name
    self.dataset_downloader = dataset_downloader
    self.checkpoint_directory = self.file_mngr.init_files_directory()
    self.checkpoint_file = self.file_mngr.join_path(self.checkpoint_directory, CHECKPOINT_FILE_NAME)
  
  # Gets checkpoint file
  def download_checkpoint(self):
    
    filename = CHECKPOINT_URL.split('/')[-1]
    filepath = self.file_mngr.join_path(self.checkpoint_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(CHECKPOINT_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(self.checkpoint_directory)
  
  # Define checkpoint
  def _set_checkpoint(self):
    
    FLAGS.checkpoint_path = self.file_mngr.join_path(self.checkpoint_directory, CHECKPOINT_FILE_NAME)
    FLAGS.checkpoint_exclude_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits/Logits'
    FLAGS.trainable_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits/Logits'
  
  # Define hyper - parameters
  def _set_hyper_parameters(self):
    
    FLAGS.max_number_of_steps = 2000
    FLAGS.batch_size = 32
    FLAGS.learning_rate = 0.001
    FLAGS.learning_rate_decay_type = 'fixed'
    FLAGS.save_interval_secs = 60
    FLAGS.save_summaries_secs = 60
    FLAGS.log_every_n_steps = 100
    FLAGS.optimizer = 'rmsprop'
    FLAGS.weight_decay = 0.00004
    
  # Prepares flowers for inception
  def define_training_parameters(self):
    
    self.file_mngr.get_or_init_training_set()
    if not os.path.exists(self.checkpoint_file):
      self.download_checkpoint()
    FLAGS.train_dir = self.file_mngr.init_files_directory()
    FLAGS.dataset_name = self.dataset_name
    FLAGS.dataset_split_name = 'train'
    FLAGS.dataset_dir = self.file_mngr.get_dataset_dir()
    
    self._set_checkpoint()
    self._set_hyper_parameters()
    
    archive_dir = self.file_mngr.get_archives_directory()
    self.dataset_downloader.run(FLAGS.dataset_dir, archive_dir)
    
  # Prepares evaluation parameters
  def define_eval_parameters(self):
  
    EVAL_FLAGS.checkpoint_path = self.file_mngr.init_files_directory()
    EVAL_FLAGS.eval_dir = self.file_mngr.get_or_init_eval_path()
    EVAL_FLAGS.dataset_name = self.dataset_name
    EVAL_FLAGS.dataset_split_name = 'validation'
    EVAL_FLAGS.dataset_dir = self.file_mngr.get_dataset_dir()
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
        
  # Train or eva;luate parameters
  def train_or_eval_net(self, args):
    
    if len(args) > 1:
      self.train_or_eval(args)
    else:
      self.train_net()
