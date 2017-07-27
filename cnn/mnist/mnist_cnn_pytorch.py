"""
Created on Jul 27, 2017

Implementation of MNIST classifier on PyTorch framework

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
  """Network model for classification"""
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    """Forward pass implementation for network model
      Args:
        x - input image
      Returns:
        result - classification result
    """
    
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    logits = self.fc2(x)
    result = F.log_softmax(logits)
    
    return result
  
def init_network(flags):
  """Initializes network model
    Args:
      flags - configuration flags
    Returns:
      model - network model
  """
  model = Net()
  if flags.cuda:
    model.cuda()
  
  return model

def init_optimizer(flags, model):
  """Initializes model optimizer
    Args:
      flags - configuration flags
      model - network model
  """
  return optim.SGD(model.parameters(), lr=flags.lr, momentum=flags.momentum)


def init_data_loaders(flags):
  """Initializes training and test data loaders
    Args:
      flags - comand line configuration flags
    Returns:
      tuple of -
        traing_loader - training data loader
        test_loader - test data loader
  """
  
  torch.manual_seed(flags.seed)
  if flags.cuda:
      torch.cuda.manual_seed(flags.seed)
  
  kwargs = {'num_workers': 1, 'pin_memory': True} if flags.cuda else {}
  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=flags.batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=flags.batch_size, shuffle=True, **kwargs)
  
  return (train_loader, test_loader)

def train(train_config):
  """Trains network model on MNIST data set
    Args:
      train_config - training configuration tuple
  """
  
  (flags, epoch, model, optimizer, train_loader) = train_config
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if flags.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % flags.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))


def test(test_config):
  """Test trained model on MNIST test set
    Args:
      test_config - test configuration tuple
  """
  
  (flags, model, test_loader) = test_config
  
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:
    if flags.cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def parse_args():
  """Parses command line configuration parameters
    Returns:
      flags - configuration parameters
  """
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size',
                      type=int,
                      default=64,
                      metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size',
                      type=int,
                      default=1000,
                      metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs',
                      type=int,
                      default=10,
                      metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr',
                      type=float,
                      default=0.01,
                      metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum',
                      type=float,
                      default=0.5,
                      metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda',
                      action='store_true',
                      default=False,
                      help='disables CUDA training')
  parser.add_argument('--cuda',
                      action='store_false',
                      default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed',
                      type=int,
                      default=1,
                      metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval',
                      type=int,
                      default=10,
                      metavar='N',
                      help='how many batches to wait before logging training status')
  flags = parser.parse_args()
  
  return flags

if __name__ == '__main__':
  """Train network on MNIST data"""
  
  flags = parse_args()
  model = init_network(flags)
  (train_loader, test_loader) = init_data_loaders(flags)
  optimizer = init_optimizer(flags, model)
  
  for epoch in range(1, flags.epochs + 1):
    train_config = (flags, epoch, model, optimizer, train_loader)
    test_config = (flags, model, test_loader)
    train(train_config)
    test(test_config)