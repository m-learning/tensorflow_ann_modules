"""
Created on Dec 27, 2016

Reads command line argument for evaluation

@author: Levan Tsinadze
"""

import argparse
import os

from cnn.cifar import network_config as network
from cnn.cifar.cnn_files import training_file


def parse_and_retrieve(batch_size=128, num_examples=10000):
  """Parses command line arguments
    Args:
      batch_size - training batch size
      num_examples - number of examples to run
    Returns:
      FLAGS - command line arguments and flags object
  """
  
  __files = training_file()
  __default_data_dir = __files.get_training_directory()
  global FLAGS
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--file_path',
                          type=str,
                          default=os.path.join(__default_data_dir, 'test_batch.bin'),
                          help='File or directory path for evaluation') 
  arg_parser.add_argument('--eval_dir',
                          type=str,
                          default=__files.init_logs_directory(),
                          help='Directory where to write event logs.')
  arg_parser.add_argument('--eval_data',
                          type=str,
                          default='test',
                          help='Either "test" or "train_eval".')
  arg_parser.add_argument('--checkpoint_dir',
                          type=str,
                          default=__files.init_files_directory(),
                          help='Directory where to read model checkpoints.')
  arg_parser.add_argument('--eval_interval_secs',
                          type=int,
                          default=60 * 5,
                          help='How often to run the eval.')
  arg_parser.add_argument('--num_examples',
                          type=int,
                          default=num_examples,
                          help='Number of examples to run.')
  arg_parser.add_argument('--run_once',
                          dest='run_once',
                          action='store_true',
                          help='Whether to run eval only once.')
  arg_parser.add_argument('--not_run_once',
                          dest='run_once',
                          action='store_false',
                          help='Whether to run eval not only once.')
  arg_parser.add_argument('--batch_size',
                          type=int,
                          default=batch_size,
                          help='Number of images to process in a batch.')
  arg_parser.add_argument('--data_dir',
                          type=str,
                          default=__default_data_dir,
                          help='Path to the CIFAR-10 data directory.')
  arg_parser.add_argument('--use_fp16',
                          dest='use_fp16',
                          action='store_true',
                          help='Train the model using fp16.')
  arg_parser.add_argument('--not_use_fp16',
                          dest='use_fp16',
                          action='store_false',
                          help='Train the model using fp32.')
  (FLAGS, _) = arg_parser.parse_known_args()
  network.FLAGS = FLAGS
  
  return FLAGS
