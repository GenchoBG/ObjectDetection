import os

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans as KMeans
from pyclustering.utils.metric import type_metric, distance_metric

from ObjectDetectionElsys.utils import parse_annotation, calculate_IoU, Object


def distance(boxcoords, centroid):
    if boxcoords.ndim > 1:
        result = []

        for i in range(len(boxcoords)):
            result.append(distance(boxcoords[i], centroid[i]))

        return result

    box = Object(xmin=0, xmax=boxcoords[0], ymin=0, ymax=boxcoords[1])

    centroid_box = Object(xmin=0, xmax=centroid[0], ymin=0, ymax=centroid[1])

    return 1 - calculate_IoU(box, centroid_box)


def find_anchor_boxes(kmeans_values, n_boxes):
    initial_centers = kmeans_plusplus_initializer(kmeans_values, n_boxes).initialize()

    metric = distance_metric(type_metric.USER_DEFINED, func=distance)

    kmeans = KMeans(n_clusters=n_boxes, data=kmeans_values, initial_centers=initial_centers, metric=metric)

    kmeans.process()

    centers = kmeans.get_centers()
    clusters = kmeans.get_clusters()

    centers.sort(key=lambda c: c[0])

    return centers, clusters


def get_average_IoU(kmeans_values, centers):
    max_IoUs = []

    for obj in kmeans_values:
        IoUs = []
        for center in centers:
            current_IoU = 1 - distance(obj, center)
            IoUs.append(current_IoU)

        max_IoU = max(IoUs)
        max_IoUs.append(max_IoU)

    average_IoU = sum(max_IoUs) / len(max_IoUs)

    return average_IoU

def get_kmeans_values_from_annotations(annotations_path, grid_width, grid_height):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(annotations_path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))

    #files = files[:500]

    print(f'Found {len(files)} annotation files')
    annotations = [parse_annotation(file) for file in files]

    kmeans_values = []
    for annotation in annotations:
        cell_width = annotation.imagewidth / grid_width
        cell_height = annotation.imageheight / grid_height

        for obj in annotation.objects:
            width = (obj.xmax - obj.xmin) / cell_width
            height = (obj.ymax - obj.ymin) / cell_height

            kmeans_values.append(np.array([width, height]))

    return kmeans_values

def format_anchors(anchors):
    anchors = np.array(anchors)
    anchors = anchors.ravel()
    anchors = [anchor.round(2) for anchor in anchors]

    return anchors
