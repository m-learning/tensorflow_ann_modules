'''
Created on Sep 23, 2016

Evaluation parameters for CNN model

@author: Levan Tsinadze
'''

# Flags to evaluate model
batch_size = 100  # 'The number of samples in each batch.'

max_num_batches = None  # 'Max number of batches to evaluate by default use all.'

master = ''  # 'The address of the TensorFlow master to use.'

checkpoint_path = '/tmp/tfmodel/'
    # 'The directory where the model was written to or an absolute path to a '
    # 'checkpoint file.'

eval_dir = '/tmp/eval/'  # 'Directory where the results are saved to.'

num_preprocessing_threads = 4  # 'The number of threads used to create the batches.'

dataset_name = 'imagenet'  # 'The name of the dataset to load.'

dataset_split_name = 'test'  # 'The name of the train/test split.'

dataset_dir = None  # 'The directory where the dataset files are stored.'

labels_offset = 0
    # 'An offset for the labels in the dataset. This flag is primarily used to '
    # 'evaluate the VGG and ResNet architectures which do not use a background '
    # 'class for the ImageNet dataset.'

model_name = 'inception_resnet_v2'  # 'The name of the architecture to evaluate.'

preprocessing_name = None  # 'The name of the preprocessing to use. If left '
    # 'as `None`, then the model_name flag is used.'

moving_average_decay = None
    # 'The decay to use for the moving average.'
    # 'If left as None, then moving averages are not used.'

eval_image_size = None  # 'Eval image size'

network_name = 'vgg_16'  # Name of network

layer_key = 'vgg_16/fc8'  # Layer to call for prediction
