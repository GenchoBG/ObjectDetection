from utils import get_annotations_images
import shutil
from PIL import Image as Img
import numpy as np
import random
from augmentation import read_image


def augment_images(cfg, images_folder, annotation_folder, augmenters, target_count, max_augs=3, target_ann_folder=None,
                   target_im_folder=None):
    annotations_files, images = get_annotations_images(annotation_folder, images_folder)

    current_count = len(annotations_files)

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

def main():
    #cfg
    #augment
    pass

main()
