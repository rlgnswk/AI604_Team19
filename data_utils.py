import torch 
import torch.nn as nn #
from torch.utils import data
import torchvision


def dataAug(gt,lr):

  '''
  do data augmentation like rotate, flip ..
  '''

  return gt, lr



class Dataset(Dataset):
    def __init__(self):
      pass
    def __getitem__(self, index):
      pass
    def __len__(self):
      pass


def Load_Data():

    '''expected file structure:


    main/dataset - trainimage
                 - testimage
                 - validimage     
    '''


    return trainDataLoader, validDataLoader , testDataLoader
