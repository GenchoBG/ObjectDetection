from utils import get_annotations_images
import shutil
from PIL import Image as Img
import numpy as np
import random
from augmentation import read_image, display_image, change_brightness_slightly, change_brightness_not_so_slightly, dropout, adjust_contrast, grayscale, noise, blur, sharpen
import argparse
from config import Config

def augment_images(cfg, images_folder, annotation_folder, augmenters, target_count, max_augs=3, target_ann_folder=None,
                   target_im_folder=None):
    annotations_files, images = get_annotations_images(annotation_folder, images_folder)

    current_count = len(annotations_files)
    if current_count >= target_count:
        print(f'Warning: Current count is already >= target ({current_count})')


    if target_ann_folder == None:
        target_ann_folder = annotation_folder
    else:
        for i in range(current_count):
            shutil.copy(annotations_files[i], f'{target_ann_folder}\\{i}.xml')

    if target_im_folder == None:
        target_im_folder = images_folder
    else:
        for i in range(current_count):
            shutil.copy(images[i], f'{target_im_folder}\\{i}.jpg')

    ground_truth_count = len(annotations_files)
    while (current_count < target_count):
        i = int(random.uniform(0, ground_truth_count))

        im = read_image(images[i], (cfg.get('image_width'), cfg.get('image_height')))

        augs = int(random.uniform(0, max_augs)) + 1
        for aug in random.choices(augmenters, k=augs):
            im = aug(im)

        im = im.astype(np.uint8)
        im = Img.fromarray(im, 'RGB')
        im.save(f'{target_im_folder}\\{current_count}.jpg')

        shutil.copy(annotations_files[i], f'{target_ann_folder}\\{current_count}.xml')

        current_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', metavar='config', type=str, default=None)
    parser.add_argument('--images', metavar='images', type=str, default = './images/')
    parser.add_argument('--annotations', metavar='annotations', type=str, default='./annotations/')
    parser.add_argument('--target', metavar='target_count', type=int)
    parser.add_argument('--max_augs', metavar='max_augs', type=int, default=3)

    parser.add_argument('--dest_ann', metavar='destination_annotations', type=str, default=None)
    parser.add_argument('--dest_im', metavar='destination_images', type=str, default=None)

    augmenters = [blur, adjust_contrast, change_brightness_not_so_slightly, change_brightness_slightly, dropout]

    args = parser.parse_args()

    cfg_path = args.cfg
    target = args.target

    if cfg_path == None or target == None:
        print('Specify target image count and config path')
        exit(1)

    images = args.images
    annotations = args.annotations
    max_augs = args.max_augs
    dest_ann = args.dest_ann
    dest_im = args.dest_im

    cfg = Config(cfg_path)

    augment_images(cfg, images, annotations, augmenters, target, max_augs, dest_ann, dest_im)
