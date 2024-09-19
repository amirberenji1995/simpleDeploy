import torch

def scaler(arr):
    
  mu = arr.float().mean(dim = 1)
  sig = arr.float().std(dim = 1)

  array_scaled = (torch.subtract(arr.transpose(0, 1), mu) / sig).transpose(0, 1)

  return array_scaled