from IPython.core.display import Image, display
from PIL import Image as Img
from PIL import ImageDraw as ImgDraw
from PIL import ImageFont
from random import uniform
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
from json import JSONEncoder

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

    def supports(self, label):
        return label in self.__dict__

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

class MyJSONEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

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
                    
        if(curr.difficult == 0):
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

font = ImageFont.load_default()
#font = ImageFont.truetype("./fonts/comic-sans-ms/COMIC.TTF", 14)

def _get_color():
    return (0, 255, 0)
    return (int(uniform(0, 255)), int(uniform(0, 255)), int(uniform(0, 255)))


def draw_image(imagepath, objects=[], draw_grid=False, grid_size=(0, 0), save=False, displ = True):
    im = Img.open(imagepath)
    draw = ImgDraw.Draw(im)

    if draw_grid:
        width_factor = im.width / grid_size[0]
        height_factor = im.height / grid_size[1]

        for i in range(max(grid_size[0], grid_size[1])):
            draw.line((i * width_factor, 0) + (i * width_factor, im.height), fill=0, width=1)
            draw.line((0, i * height_factor) + (im.width, i * height_factor), fill=0, width=1)

    for obj in objects:
        # print(obj)
        color = _get_color()

        draw.text((obj.xmin, obj.ymin - 20), f"{obj.name}: {round(obj.conf, 2)}", font=font, fill=color)
        draw.line((obj.xmin, obj.ymin) + (obj.xmax, obj.ymin), fill=color, width=2)
        draw.line((obj.xmin, obj.ymax) + (obj.xmax, obj.ymax), fill=color, width=2)
        draw.line((obj.xmin, obj.ymin) + (obj.xmin, obj.ymax), fill=color, width=2)
        draw.line((obj.xmax, obj.ymin) + (obj.xmax, obj.ymax), fill=color, width=2)


        # xmid = (obj.xmax + obj.xmin) / 2
        # ymid = (obj.ymax + obj.ymin) / 2
        # draw.line((xmid, ymid) + (xmid, ymid), fill = color, width = 2)
    if save:
        path = imagepath.replace('\\', r'/')
        filename = path.split('/')[-1]
        im.save(f'./out/{filename}')

    if displ:
        display(im)

def save_objects_to_json(imagepath, objects):
    filename = imagepath.split('/')[-1]

    text_file = open(f'./out/{filename}.json', "w")
    n = text_file.write(json.dumps(objects, cls=MyJSONEncoder))
    text_file.close()

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

def image_to_mobilenet_input(imagepath, inputshape):
    im = Img.open(imagepath).resize(inputshape)
    im = np.array(im, np.float32)
    im /= 255
    im -= 0.5
    im *= 2.

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


def get_annotations_images(annotations_dir, images_dir):
    image_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']

    annons, images = [], []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(annotations_dir):
        for file in f:
            if '.xml' in file:
                image_exists = False
                for im_format in image_formats:
                    image_name = file[:-4] + im_format

                    image_path = images_dir + '\\' + image_name

                    if os.path.exists(image_path):
                        image_exists = True
                        break

                if image_exists:
                    annons.append(annotations_dir + '\\' + file)
                    images.append(images_dir + '\\' + image_name)

    # annons = annons[:250]
    # images = images[:250]

    return annons, images

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

