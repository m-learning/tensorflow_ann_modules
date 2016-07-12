'''
Created on Jun 28, 2016

Runs retrained neural network for recognition

@author: Levan Tsinadze
'''

from cnn.transfer.general_recognizer import retrained_recognizer
from cnn.zebras.cnn_files import training_file

# Recognizes image thru trained neural networks
class image_recognizer(retrained_recognizer):
  
  def __init__(self):
    tr_file = training_file()
    super(image_recognizer, self).__init__(tr_file)

# Runs image recognition
if __name__ == '__main__':
  img_recognizer = image_recognizer()
  img_recognizer.run_inference_on_image()
