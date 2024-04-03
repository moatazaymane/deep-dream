import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import copy
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from utils import clip
import os
import cv2
import IPython


class ClassImage:


  def __init__(self, chosen_class=130, iterations=100, learning_rate=0.8):

    self.model = models.alexnet(pretrained=True)

    #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="Inception_V3_Weights.IMAGENET1K_V1")
    self.chosen_class = chosen_class
    self.iterations = iterations
    self.learning_rate = learning_rate
    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]

  def get_class_image(self):
      
      if not os.path.isdir(os.getcwd() + "/class_images_results"):
         os.makedirs(os.getcwd() + "/class_images_results")

      noise_image = ClassImage.init_image()
      _, dep = self.gradient_ascent_loop(self.iterations, noise_image, with_optimizer=True)

      
  def gradient_ascent_step(self, image, iteration):

    # Running gradient descent on -1*target_class_logit
    loss = -self.model(image)[:, self.chosen_class]

    gradients = torch.autograd.grad(loss, image)
    image = image - self.learning_rate*gradients[0]

    if not (iteration+1)%10:
      print(f"Logit after ascent iteration {iteration + 1} : {-loss}")
      dep = self.deprocess_image(image)
      print(dep.shape)
      pil_image = Image.fromarray(self.deprocess_image(image)).resize((200,200))
      IPython.display.display(pil_image)

    return image


  def gradient_ascent_step_optim(self, loss, image, iteration, optimizer):

    # Running gradient descent on -1*target_class_logit
    image_prev = image.clone()

    self.model.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.max(image_prev-image).detach().numpy().tolist() != 0

    if not (iteration+1)%10:

      print(f"Logit after ascent iteration {iteration + 1} : {-loss.item():.2f}")
      dep = copy.copy(image)
      dep = self.deprocess_image(dep)
      ClassImage.display_images(dep, self.chosen_class)

    return image, loss



  def gradient_ascent_loop(self, steps, noise_image, with_optimizer=False):

      ff = noise_image.copy()
      preprocessed_image = self.preprocess_image(noise_image)
      optimizer = torch.optim.SGD([preprocessed_image], lr=self.learning_rate, weight_decay=2) #2*lambda(L2)

      initial_noise = preprocessed_image.clone()
      for iteration in tqdm(range(self.iterations)):
        if with_optimizer:
            loss = -self.model(preprocessed_image)[:, self.chosen_class]
            preprocessed_image, loss = self.gradient_ascent_step_optim(loss, preprocessed_image, iteration, optimizer)
            if (iteration+1)%10 == 0:
              print("Image change MAD:")
              inte = torch.subtract(initial_noise,preprocessed_image).detach().apply_(lambda x: abs(x)).sum()/(preprocessed_image.shape[1]*preprocessed_image.shape[2]*preprocessed_image.shape[3])
              print(inte.item())

        else:
          preprocessed_image = self.gradient_ascent_step(preprocessed_image, iteration)

      deprocessed_image = self.deprocess_image(preprocessed_image)

      assert noise_image.shape == deprocessed_image.shape
      assert np.max(ff - deprocessed_image).tolist()!=0
      return preprocessed_image, deprocessed_image


  def display_image(self, image):

    im = copy.copy(image)
    im =  self.deprocess_image(im)
    pil_image = Image.fromarray(im)
    #IPython.display.display(pil_image)


  @staticmethod
  def display_images(image, c_class):

    class_image = copy.copy(image)
    lab = cv2.cvtColor(class_image, cv2.COLOR_BGR2LAB)
    out_path = os.getcwd() + f"/class_image_results/class_image_{c_class}.jpeg"
    print(f"Writing image into file {out_path}")
    cv2.imwrite(os.getcwd() + f"/class_images_results/class_image_{c_class}.jpeg",lab)
    a_component = lab[:,:,1]
    th = cv2.threshold(a_component,140,255,cv2.THRESH_BINARY)[1]
    th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    heatmap_img = cv2.applyColorMap(class_image, cv2.COLORMAP_JET)
    #IPython.display.display(np.hstack((class_image,th, heatmap_img)))


  @staticmethod
  def init_image():
      noise_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
      return noise_image

  def preprocess_image(self, image):

    image = image.astype(np.float32).transpose(2, 0, 1)

    for channel in range(3):
      image[channel]/=255
      image[channel]-=self.mean[channel]
      image[channel]/=self.std[channel]

    image = torch.from_numpy(image).float().unsqueeze(dim=0)

    return Variable(image, requires_grad=True)

  def deprocess_image(self, processed_image):

    dep = copy.copy(processed_image)
    dep = dep.detach().numpy()[0]

    for channel in range(3):
      dep[channel]*=self.std[channel]
      dep[channel]+=self.mean[channel]

    dep*=255
    dep = np.clip(dep, 0, 255)
    return np.uint8(dep).transpose(1, 2, 0)



  @staticmethod
  def diplay_image(image):
      Image.fromarray(image)



  def gradient_ascent_loop(self, steps, noise_image, with_optimizer=False):

      ff = noise_image.copy()
      preprocessed_image = self.preprocess_image(noise_image)
      optimizer = torch.optim.SGD([preprocessed_image], lr=self.learning_rate, weight_decay=2) #2*lambda(L2)

      initial_noise = preprocessed_image.clone()
      for iteration in tqdm(range(self.iterations)):
        if with_optimizer:
            loss = -self.model(preprocessed_image)[:, self.chosen_class]
            preprocessed_image, loss = self.gradient_ascent_step_optim(loss, preprocessed_image, iteration, optimizer)
            if (iteration+1)%10 == 0:
              print("Image change MAD:")
              inte = torch.subtract(initial_noise,preprocessed_image).detach().apply_(lambda x: abs(x)).sum()/(preprocessed_image.shape[1]*preprocessed_image.shape[2]*preprocessed_image.shape[3])
            preprocessed_image.data = clip(preprocessed_image.data)
        else:
          preprocessed_image = self.gradient_ascent_step(preprocessed_image, iteration)

      deprocessed_image = self.deprocess_image(preprocessed_image)

      assert noise_image.shape == deprocessed_image.shape
      assert np.max(ff - deprocessed_image).tolist()!=0
      return preprocessed_image, deprocessed_image


  def display_image(self, image):

    im = copy.copy(image)
    im =  self.deprocess_image(im)
    pil_image = Image.fromarray(im)
    IPython.display.display(pil_image)



  @staticmethod
  def init_image():
      noise_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
      return noise_image

  def preprocess_image(self, image):

    image = image.astype(np.float32).transpose(2, 0, 1)

    for channel in range(3):
      image[channel]/=255
      image[channel]-=self.mean[channel]
      image[channel]/=self.std[channel]

    image = torch.from_numpy(image).float().unsqueeze(dim=0)

    return Variable(image, requires_grad=True)

  def deprocess_image(self, processed_image):

    dep = copy.copy(processed_image)
    dep = dep.detach().numpy()[0]

    for channel in range(3):
      dep[channel]*=self.std[channel]
      dep[channel]+=self.mean[channel]

    dep*=255
    dep = np.clip(dep, 0, 255)
    return np.uint8(dep).transpose(1, 2, 0)


  @staticmethod
  def diplay_image(image):
      Image.fromarray(image)


if __name__ == '__main__':
   import argparse

   argparser = argparse.ArgumentParser()

   # 55: Green Snake
   argparser.add_argument('--c_class', default=55)
   class_image = ClassImage(chosen_class=int(argparser.parse_args().c_class))
   class_image.get_class_image()


   
   