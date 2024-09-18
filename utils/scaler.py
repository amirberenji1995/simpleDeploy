# import numpy as np

# def scaler(arr):
    
#   array_mu = np.mean(arr, axis = 1)
#   array_sig = np.std(arr, axis = 1)

#   array_scaled = (np.subtract(arr.transpose(), array_mu) / array_sig).transpose()

#   return array_scaled


import torch

def scaler(arr):
    
  mu = arr.float().mean(dim = 1)
  sig = arr.float().std(dim = 1)

  array_scaled = (torch.subtract(arr.transpose(0, 1), mu) / sig).transpose(0, 1)

  return array_scaled