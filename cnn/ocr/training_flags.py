"""
Created on Feb 28, 2017

Flags for model training and evaluation

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime

from cnn.ocr import network_config as config
from cnn.ocr import network_model as network


def init_network_parameters(img_w):
  """Initializes network parameters
    Args:
      img_w - image width
    Returns:
      tuple of -
        model parameters - tuple of
          model - network model
          input_data - network input data
        training parameters - tuple of
          y_pred - prediction label
          img_gen - OCR image generator
  """
  
  img_gen = config.init_img_gen(img_w)
  (input_data, model, y_pred) = network.init_model(img_w, img_gen, config.ctc_lambda_func)
  
  return ((model, input_data), (y_pred, img_gen))

def parse_arguments():
  """Parses command line arguments
    Returns:
      args - parsed command line arguments
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_width',
                      type=int,
                      default=128,
                      help='Input image width')
  parser.add_argument('--second_phase_width',
                      type=int,
                      default=512,
                      help='Input image width for seconf phase of training')
  parser.add_argument('--start_epoch',
                      type=int,
                      default=0,
                      help='Training start epoch')
  parser.add_argument('--stop_epoch',
                      type=int,
                      default=20,
                      help='Training stop epoch')
  parser.add_argument('--stop_second_phase',
                      type=int,
                      default=25,
                      help='Training stop epoch for seconf phase')
  parser.add_argument('--run name',
                      type=str,
                      default=datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                      help='Training run name')
  (args, _) = parser.parse_known_args()
  
  return args
