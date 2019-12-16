from IPython.core.display import Image, display
from PIL import Image as Img
from PIL import ImageDraw as ImgDraw
from random import uniform
import xml.etree.ElementTree as ET
import numpy as np

def read_labels(path):    
    with open(path) as f:
        labels = f.readlines()
    
    labels = [l.strip() for l in labels] 
    labels_count = len(labels)
    
    return labels, labels_count

class LabelEncoder():
    def __init__(self, labels):
        self.__dict__ = dict()
        
        index = 0
        for label in labels:
            self.__dict__[label] = index
            self.__dict__[index] = label
            index += 1
            
    def encode(self, label):
        return self.__dict__[label]
    
    def decode(self, index):
        return self.__dict__[index]

class Object(object):
    def __init__(self, xmin = None, ymin = None, xmax = None, ymax = None, conf = 0, name = 'unnamed'):        
        self.name = name
        self.conf = conf
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    def __str__(self):
        return f'{self.name} ({self.conf}) ({self.xmin}, {self.ymin}) ({self.xmax}, {self.ymax})'
    
class Annotation(object):
    def __init__(self):
        self.objects = []
        self.imagewidth = None
        self.imageheight = None
        self.filename = None

def parse_annotation(filepath):    
    objects = []
    
    #TODO: rework - find objects instead of iterating
    et = ET.parse(filepath)
    for obj in et.findall('object'):
        curr = Object()
        skip = 0
        for child in obj.iter():
            if skip > 0:
                skip-=1
                continue
            if child.tag == 'part':
                skip = 6
            if child.tag != 'bndbox':
                if(child.text.isdigit()):
                    setattr(curr, child.tag, int(child.text))
                else:
                    setattr(curr, child.tag, child.text)
        objects.append(curr)
        
    
    filename = et.find('filename').text
    width = et.find('size/width').text
    height = et.find('size/height').text
    
    annotation = Annotation()
    annotation.objects = objects
    annotation.imagewidth = int(width)
    annotation.imageheight = int(height)
    annotation.filename = filename
    
    return annotation

def _get_color():
    return (int(uniform(0, 255)), int(uniform(0, 255)), int(uniform(0, 255)))

def draw_image(imagepath, objects = [], draw_grid = False, grid_size = (0, 0)):
    im = Img.open(imagepath)    
    draw = ImgDraw.Draw(im)
    
    for obj in objects:
        color = _get_color()
        
        draw.line((obj.xmin, obj.ymin) + (obj.xmax, obj.ymin), fill = color, width = 2)
        draw.line((obj.xmin, obj.ymax) + (obj.xmax, obj.ymax), fill = color, width = 2)
        draw.line((obj.xmin, obj.ymin) + (obj.xmin, obj.ymax), fill = color, width = 2)
        draw.line((obj.xmax, obj.ymin) + (obj.xmax, obj.ymax), fill = color, width = 2)
        
        
        #xmid = (obj.xmax + obj.xmin) / 2
        #ymid = (obj.ymax + obj.ymin) / 2
        #draw.line((xmid, ymid) + (xmid, ymid), fill = color, width = 2)
    
    
    if draw_grid:
        width_factor = im.width / grid_size[0]
        height_factor = im.height / grid_size[1]
        
        for i in range(max(grid_size[0], grid_size[1])):
            draw.line((i * width_factor, 0) + (i * width_factor, im.height), fill = 0, width = 1)
            draw.line((0, i * height_factor) + (im.width, i * height_factor), fill = 0, width = 1)
    
    
    display(im)	

def image_to_vgg_input(imagepath, inputshape):
    im = Img.open(imagepath).resize(inputshape)
    im = np.array(im, np.float32)
    im -= 255 / 2
    
    return im

def image_to_yolo_input(imagepath, inputshape):
    im = Img.open(imagepath).resize(inputshape)
    im = np.array(im, np.float32)
    im /= 255
    
    return im

def calculate_IoU(ground_truth, predicted):
    # intersection rectangle
    xmin = max(ground_truth.xmin, predicted.xmin)
    ymin = max(ground_truth.ymin, predicted.ymin)
    xmax = min(ground_truth.xmax, predicted.xmax)
    ymax = min(ground_truth.ymax, predicted.ymax)
    
    interArea = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    
    groundTruthArea = (ground_truth.xmax - ground_truth.xmin + 1) * (ground_truth.ymax - ground_truth.ymin + 1)
    predictedArea = (predicted.xmax - predicted.xmin + 1) * (predicted.ymax - predicted.ymin + 1)
    
    iou = interArea / float(groundTruthArea + predictedArea - interArea)
    
    return iou
