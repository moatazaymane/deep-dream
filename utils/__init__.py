import torch
from torch.autograd import Variable
import numpy as np
import copy
from config import mean, std
from PIL import Image



def preprocess_image(image: Image):

  image = copy.copy(image)


  for channel in range(3):
    image[:,channel]/=255
    image[:,channel]-=mean[channel]
    image[:,channel]/=std[channel]

  image = torch.from_numpy(image).float()

  return Variable(image, requires_grad=True)

def clip(image_tensor):

    for c in range(3):

      image_tensor[0, c] = torch.clamp(image_tensor[0, c], -mean[c]/std[c], (1-mean[c]) / std[c])

    return image_tensor

def deprocess_image(processed_image):

  dep = copy.copy(processed_image)
  dep = dep.cpu().detach().numpy()[0]

  for channel in range(3):
    dep[channel]*=std[channel]
    dep[channel]+=mean[channel]

  dep*=255
  dep = np.clip(dep, 0, 255)
  return np.uint8(dep).transpose(1, 2, 0)