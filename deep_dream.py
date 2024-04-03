import torch
from torch.autograd import Variable
from scipy.ndimage import zoom
import numpy as np
from IPython.display import display
from PIL import Image
from utils import preprocess_image, deprocess_image, clip
from config import lr
from tqdm import tqdm

## inception : lr = 0.1
# (326, 489) -> (457, 685) -> (640, 960)


def deep_dream(Block, pil_im, iterations, octaves, octave_scale, lr):

    np_im = np.array(pil_im)
    if(np_im.shape[-1] > 3):
      np_im = np_im[:,:,:3]

    image = np.expand_dims(np_im.astype(np.float32).transpose(2, 0, 1), axis=0)


    images = [image]

    for _ in range(octaves):
      images.append(zoom(np.copy(images[-1]), (1, 1, octave_scale, octave_scale), order=1))

    prev = np.zeros_like(images[-1])

    #make a copy
    original_image = np.copy(images[0])
    lost_detail = np.zeros_like(images[-1])
    images = images[::-1]
    for num_octave, im in enumerate(images):

       dreamed_image = dream(Block, np.copy(lost_detail + im),iterations,  lr,num_octave, disp = True)

       lost_detail = dreamed_image - im

       if num_octave + 1 < len(images):
         lost_detail = zoom(lost_detail, np.array(images[num_octave + 1].shape) / np.array(lost_detail.shape), order=1)

       prev = im


    return dreamed_image


def dream(Block, im, iterations, lr, octave_num,  disp = True) -> torch.tensor:

  device = "cuda" if torch.cuda.is_available() else "cpu"

  if octave_num == 0:
    im_tensor = preprocess_image(im)
  else:
    im_tensor = Variable(torch.tensor(im), requires_grad=True)
  im_prev = torch.clone(im_tensor)
  print(f'---- Starting Image : Octave {octave_num + 1} ----- || ---- Shape {im.shape} ----')

  #display(Image.fromarray(deprocess_image(torch.clone(im_tensor.data))))
  print('\n')

  for iteration in tqdm(range(iterations)):
    Block.zero_grad()
    loss = torch.linalg.vector_norm(Block(im_tensor.to(device=device)), ord=2)
    loss.backward()

    im_tensor.data += lr*im_tensor.grad.data
    im_tensor.data = clip(im_tensor.data)
    im_tensor.grad.data.zero_()

    if disp and not (iteration + 1)%20:
      print(f"Iteration {iteration +1 } : Max Change / Pixel {torch.max(im_tensor.data - im_prev).item()} | Loss : {loss.item():.2f}")
      print(f'---- Octave {octave_num} ---- Image after iteration {iteration + 1} -----')
      #display(Image.fromarray(deprocess_image(torch.clone(im_tensor.data))))
      print('\n')

  pil_i = Image.fromarray(deprocess_image(torch.clone(im_tensor.data)))
  pil_i.save(f"{os.getcwd()}/deep_dream_results/dreamed_image.jpeg")

  return im_tensor.data.detach().numpy()


if __name__ == '__main__':
  
  import argparse
  from config import vgg19_layer, octave_scale, std, mean
  from PIL import Image, UnidentifiedImageError
  from torchvision import models 
  import os

  parser = argparse.ArgumentParser()
  parser.add_argument("--image")
  parser.add_argument("--iterations", default=1)
  parser.add_argument("--octaves", default=1)

  arguments = parser.parse_args()

  try:
    pil_im = Image.open(arguments.image)

  except (UnidentifiedImageError, FileNotFoundError):
    raise ValueError("Error in loading the image - Check the provided path and the image")
  
  vgg19 = models.vgg19(pretrained=True)
  vgg_block = vgg19.features[:vgg19_layer]
  deep_dream(vgg_block, pil_im, int(arguments.iterations), int(arguments.octaves),octave_scale, lr)

