"""
Created on Jan 27, 2017

Training DAN (Deep Averaging Network) network model on IMDB data set

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from rnn.dan import network_config as config
from rnn.dan import network_model as network
from rnn.dan import datasets
from rnn.dan.rnn_files import training_file


np.random.seed(1337)  #


def _init_weights_path():
  """Generates path for trained files
    Returns:
      _weights_path - path for trained weights file
  """
  _files = training_file()
  _model_dir = _files.model_dir
  _weights_path = _files.join_path(_model_dir, config._WEIGHTS_FILE)
  
  return _weights_path

def _save_weights(model):
  """Saves trained weights for model
    Args:
      model - network model with trained weights
  """
  
  _weights_path = _init_weights_path()
  model.save_weights(_weights_path, overwrite=True)
  print('Model weights saved in ', _weights_path)

def train(args):
  """Initializes and trains DAN network module
    Args:
      args - parsed command line arguments
  """
  
  
  model = network.dan_model()
  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  (X_train, y_train, X_test, y_test) = datasets.init_dataset()
  model.summary()
  model.fit(X_train, y_train,
            batch_size=config.batch_size,
            nb_epoch=args.epochs,
            validation_data=(X_test, y_test))
  _save_weights(model)
  
if __name__ == '__main__':
  """Run DAN network model training on IMDB data set"""
  
  args = config.parse_arguments()
  train(args)
