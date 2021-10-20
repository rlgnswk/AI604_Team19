import torch 
import torch.nn as nn #
from torch.utils import data
import torchvision


def dataAug(gt,lr):

  '''
  do data augmentation in ZSSR paper like rotate, flip ..


  return augmentated tensor gt and lr
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


    main/dataset - name of trainimage

    '''


    return trainDataLoader #, validDataLoader , testDataLoader
