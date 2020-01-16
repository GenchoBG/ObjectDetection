from utils import get_annotations_images
import os


def filter_datasaet(images_dir, annotations_dir, image_sets_dir, wanted_sets):
    wanted = set()
    # r=root, d=directories, f = files
    for r, d, f in os.walk(image_sets_dir):
        for file in f:
            for wanted_set in wanted_sets:
                if wanted_set in file:
                    print(file)
                    with open(f'{image_sets_dir}\\{file}') as image_set_file:
                        sample_files = image_set_file.readlines()
                        for line in sample_files:
                            args = line.split(' ')
                            sample_file = args[0].strip()
                            indicator = args[-1].strip()
                            if indicator == "1":
                                wanted.add(sample_file)

    annotations, images = get_annotations_images(annotations_dir, images_dir)
    wanted_annotatioins = []
    wanted_images = []
    for i in range(len(annotations)):
        ann = annotations[i].replace('\\', '/').split('/')[-1]
        ann = ann[:-4]

        if ann in wanted:
            wanted_annotatioins.append(annotations[i])
            wanted_images.append(images[i])

    return wanted_annotatioins, wanted_images

if __name__ == '__main__':
    images_dir = r'./images'
    annotations_dir = r'./annotations'
    image_sets_dir = r'./imagesets'

    wanted_sets = ['bicycle', 'bus', 'car', 'motorbike', 'person']

    annotations, images = get_annotations_images(annotations_dir, images_dir)

    wanted_annotatioins, wanted_images = filter_datasaet(images_dir, annotations_dir, image_sets_dir, wanted_sets)

    print(len(annotations))
    print(len(wanted_annotatioins))