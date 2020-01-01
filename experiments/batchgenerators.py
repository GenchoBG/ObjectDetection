import numpy as np
from augmentation import read_image

def batch_generator(annotations, images, cfg, normalizer_func, encoder_func, raw_files=True):
    ins = []
    outs = []

    while True:
        for index in range(len(images)):
            image = read_image(images[index], inputshape=(cfg.get('image_width'), cfg.get('image_height')))

            ins.append(normalizer_func(image))
            outs.append(encoder_func(annotations[index], raw_files))

            if len(ins) == cfg.get('batch_size'):
                yield (np.array(ins, dtype=np.float32), np.array(outs, dtype=np.float32))
                ins = []
                outs = []

#TODO: batch_generator_augmentation