from IPython.core.display import display
from PIL import Image as Img
import random
import numpy as np
from scipy.ndimage.filters import convolve

def read_image(path, inputshape):
    im = Img.open(path).resize(inputshape).convert('RGB')
    im = np.array(im, dtype = np.float32)

    return im

    
def normalize(imagearr):
    imagearr = np.where(imagearr >= 0, imagearr, 0)
    imagearr = np.where(imagearr <= 255, imagearr, 255)
    
    return imagearr

def display_image(imagearr): 
    imagearr = imagearr.astype(np.uint8)
    img = Img.fromarray(imagearr, 'RGB')
    display(img)
    
def change_brightness_slightly(im, range_ = (-20, 20)):
    value = random.uniform(range_[0], range_[1])
    im += value
    im = normalize(im)
    return im

def change_brightness_not_so_slightly(im, range_ = (0.5, 1.5)):
    value = random.uniform(range_[0], range_[1])
    im *= value
    im = normalize(im)
    return im

def dropout(im, chance = 0.1):
    percent = random.uniform(0, chance)
    
    shapeprod = im.shape[0] * im.shape[1] * im.shape[2]
    falses = [False for i in range(int(percent * shapeprod))]
    trues = [True for i in range(int((1 - percent) * shapeprod) + 1)]
    mask = np.array(trues + falses)
    
    np.random.shuffle(mask)
    mask = mask.reshape(im.shape)
    
    im = np.where(mask, im, 0)
    
    return im

def adjust_contrast(im, range_ = (-100, 100)):
    value = random.uniform(range_[0], range_[1])
    factor = (259 * (value + 255)) / (255 * (259 - value))

    im = 128 + factor * (im - 128)
    
    return im

def grayscale(im):
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    
    im[..., 0] = 0.2989 * r + 0.5870 * g + 0.1140 * b
    im[..., 1] = im[..., 0]
    im[..., 2] = im[..., 0]
    
    return im

def noise(im, mean = 0, sigma = 0.05):
    sigma = random.uniform(0, sigma * 255) 
    
    gaussian = np.random.normal(mean, sigma, im.shape) 
    
    return im + gaussian

box_blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
    ]) / 9
    
gaussian_blur_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
    ]) / 16

blur_kernels = [box_blur_kernel, gaussian_blur_kernel]

def blur(im):
    kernel = random.choice(blur_kernels)

    im[..., 0] = convolve(im[..., 0], kernel)
    im[..., 1] = convolve(im[..., 1], kernel)
    im[..., 2] = convolve(im[..., 2], kernel)
    
    return im

sharpen_kernel = np.array([
[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]
])

def sharpen(im):
    kernel = sharpen_kernel

    im[..., 0] = convolve(im[..., 0], kernel)
    im[..., 1] = convolve(im[..., 1], kernel)
    im[..., 2] = convolve(im[..., 2], kernel)
    
    return im
