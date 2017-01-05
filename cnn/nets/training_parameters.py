"""
Created on Sep 23, 2016

Training parameters 

@author: Levan Tsinadze
"""

master = ''  # 'The address of the TensorFlow master to use.'

train_dir = '/tmp/tfmodel/'  # 'Directory where checkpoints and event logs are written to.'

num_clones = 1  # 'Number of model clones to deploy.'

clone_on_cpu = False  # 'Use CPUs to deploy clones.'

worker_replicas = 1  # 'Number of worker replicas.'

num_ps_tasks = 0  # 'The number of parameter servers. If the value is 0, then the parameters '
                  # 'are handled locally by the worker.'

num_readers = 4  # 'The number of parallel readers that read data from the dataset.'

num_preprocessing_threads = 4  # 'The number of threads used to create the batches.'

log_every_n_steps = 10  # 'The frequency with which logs are print.'

save_summaries_secs = 600  # 'The frequency with which summaries are saved, in seconds.'

save_interval_secs = 600,  # 'The frequency with which the model is saved, in seconds.'

task = 0  # 'Task id of the replica running the training.'

######################
# Optimization Flags #
######################

weight_decay = 0.00004  # The weight decay on the model weights.'

optimizer = 'rmsprop'  # 'The name of the optimizer, one of "adadelta", "adagrad", "adam",' '"ftrl", "momentum", "sgd" or "rmsprop".'

adadelta_rho = 0.95  # 'The decay rate for adadelta.'

adagrad_initial_accumulator_value = 0.1  # 'Starting value for the AdaGrad accumulators.'

adam_beta1 = 0.9  # 'The exponential decay rate for the 1st moment estimates.'

adam_beta2 = 0.999  # 'The exponential decay rate for the 2nd moment estimates.'

opt_epsilon = 1.0  # 'Epsilon term for the optimizer.'

ftrl_learning_rate_power = -0.5  # 'The learning rate power.'

ftrl_initial_accumulator_value = 0.1  # 'Starting value for the FTRL accumulators.'

ftrl_l1 = 0.0  # 'The FTRL l1 regularization strength.'

ftrl_l2 = 0.0  # 'The FTRL l2 regularization strength.'

momentum = 0.9  # 'The momentum for the MomentumOptimizer and RMSPropOptimizer.'

rmsprop_momentum = 0.9  # 'Momentum.'

rmsprop_decay = 0.9  # Decay term for RMSProp.'

#######################
# Learning Rate Flags #
#######################

learning_rate_decay_type = 'fixed'  # 'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
                                            # ' or "polynomial"'

learning_rate = 0.01  # 'Initial learning rate.'

end_learning_rate = 0.0001  # 'The minimal end learning rate used by a polynomial decay learning rate.'

label_smoothing = 0.0  # 'The amount of label smoothing.'

learning_rate_decay_factor = 0.94  # 'Learning rate decay factor.'

num_epochs_per_decay = 2.0  # 'Number of epochs after which learning rate decays.'

sync_replicas = False  # 'Whether or not to synchronize the replicas during training.'

replicas_to_aggregate = 1  # 'The Number of gradients to collect before updating params.'

moving_average_decay = None
    # 'The decay to use for the moving average.'
    # 'If left as None, then moving averages are not used.'

#######################
# Dataset Flags #
#######################

dataset_name = 'imagenet'  # 'The name of the dataset to load.'

dataset_split_name = 'train'  # 'The name of the train/test split.'

dataset_dir = None  # The directory where the dataset files are stored.'

labels_offset = 0
    # 'An offset for the labels in the dataset. This flag is primarily used to '
    # 'evaluate the VGG and ResNet architectures which do not use a background '
    # 'class for the ImageNet dataset.'

model_name = 'vgg_16'  # 'The name of the architecture to train.')

preprocessing_name = None  # 'The name of the preprocessing to use. If left '
    # 'as `None`, then the model_name flag is used.'

batch_size = 32  # 'The number of samples in each batch.'

train_image_size = None  # 'Train image size'

max_number_of_steps = None  # 'The maximum number of training steps.'

#####################
# Fine-Tuning Flags #
#####################

checkpoint_path = None  # 'The path to a checkpoint from which to fine-tune.'

checkpoint_exclude_scopes = 'vgg16/fc7,vgg16/fc8'
    # 'Comma-separated list of scopes of variables to exclude when restoring '
    # 'from a checkpoint.'

trainable_scopes = 'vgg16/fc7,vgg16/fc8'
    # 'Comma-separated list of scopes to filter the set of variables to train.'
    # 'By default, None would train all the variables.'

ignore_missing_vars = False  # 'When restoring a checkpoint would ignore missing variables.'

network_name = 'vgg_16'  # Name of network

layer_to_train = ('fc7', 'fc8')
