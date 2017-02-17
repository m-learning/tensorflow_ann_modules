"""
Created on Feb 18, 2017

Module for OCR model training

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
from keras.optimizers import SGD

from cnn.ocr.image_ocr_keras import VizCallback
from cnn.ocr.network_config import OUTPUT_DIR
from cnn.ocr.network_config import words_per_epoch, val_words


def init_sgd_optimizer():
  """Initializes stochastic gradient descend optimizer"""
  
  return SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

def prepare_training(model, train_parameters):
  """Prepares model for training
    model - network model
    train_parameters - training parameters
  """
  
  (run_name, start_epoch, _) = train_parameters
  sgd = init_sgd_optimizer()
  model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
  if start_epoch > 0:
    weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
    model.load_weights(weight_file)
  
def train_model(model, input_data, y_pred, img_gen, train_parameters):
  
  (run_name, start_epoch, stop_epoch) = train_parameters
  test_func = K.function([input_data], [y_pred])

  viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

  model.fit_generator(generator=img_gen.next_train(), samples_per_epoch=(words_per_epoch - val_words),
                      nb_epoch=stop_epoch, validation_data=img_gen.next_val(), nb_val_samples=val_words,
                      callbacks=[viz_cb, img_gen], initial_epoch=start_epoch)
