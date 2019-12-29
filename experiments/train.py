import tensorflow as tf
import numpy as np
from utils import parse_annotation, Object, calculate_IoU, LabelEncoder
from config import Config
import random


class YOLO():
    def __init__(self, cfg, encoder, networkfactory):
        self.cfg = cfg
        self.encoder = encoder
        self.networkfactory = networkfactory

    def encode_y_true_from_annotation(self, annotation, raw_file=True):
        y_true = np.zeros(shape=(self.cfg.get('grid_width'), self.cfg.get('grid_height'),
                                 self.cfg.get('boxes'), 5 + self.cfg.get('classes')))
        objs = [[[] for col in range(self.cfg.get('grid_width'))] for row in range(self.cfg.get('grid_height'))]

        if raw_file:
            annotation = parse_annotation(annotation)

        image_cell_width = annotation.imagewidth / self.cfg.get('grid_width')
        image_cell_height = annotation.imageheight / self.cfg.get('grid_height')

        for obj in annotation.objects:
            obj.xmid = (obj.xmax + obj.xmin) / 2
            obj.ymid = (obj.ymax + obj.ymin) / 2
            obj.width = obj.xmax - obj.xmin
            obj.height = obj.ymax - obj.ymin

            row = int(obj.ymid / image_cell_height)
            col = int(obj.xmid / image_cell_width)
            # print(f'row {row} col {col}')

            objs[row][col].append(obj)

        for row in range(self.cfg.get('grid_height')):
            for col in range(self.cfg.get('grid_width')):
                cell_objs = objs[row][col]
                random.shuffle(cell_objs)

                for obj in cell_objs:
                    best_anchor_index = 0
                    best_IoU = -1

                    for index in range(self.cfg.get('boxes')):
                        anchor_w, anchor_h = self.cfg.get('anchors')[index]
                        width = anchor_w * image_cell_width
                        height = anchor_h * image_cell_height

                        xmid = (col + 0.5) * image_cell_width
                        ymid = (row + 0.5) * image_cell_height

                        anchor_object = Object(xmin=xmid - width / 2, xmax=xmid + width / 2, ymin=ymid - height / 2,
                                               ymax=ymid + height / 2)

                        current_IoU = calculate_IoU(obj, anchor_object)

                        if current_IoU > best_IoU:
                            best_IoU = current_IoU
                            best_anchor_index = index

                    '''
                    grid_center_x = (row + 0.5) * image_cell_width
                    grid_center_y = (col + 0.5) * image_cell_height
    
                    x = obj.xmid - grid_center_x
                    y = obj.ymid - grid_center_y
    
                    w = np.log(obj.width / anchors[best_anchor_index][0]) 
                    h = np.log(obj.height / anchors[best_anchor_index][1]) 
                    '''
                    x = obj.xmid
                    y = obj.ymid

                    w = obj.width
                    h = obj.height

                    c = 1  # best_IoU

                    detector = np.zeros(shape=(5 + self.cfg.get('classes')))

                    detector[0] = c
                    detector[1] = x
                    detector[2] = y
                    detector[3] = w
                    detector[4] = h

                    label_index = self.encoder.encode(obj.name)
                    detector[5 + label_index] = 1

                    y_true[row][col][best_anchor_index] = detector

        return y_true


    def custom_loss(self, y_true, y_pred):
        c_pred = y_pred[:, :, :, :, 0]
        c_true = y_true[:, :, :, :, 0]

        c_pred = tf.sigmoid(c_pred)


        output_shape = (self.cfg.get('batch_size'), self.cfg.get('grid_width'), self.cfg.get('grid_height'), self.cfg.get('boxes'))

        cell_x = tf.to_float(
            tf.reshape(tf.keras.backend
                        .repeat_elements(
                            tf.tile(tf.range(self.cfg.get('grid_width')),
                                    [self.cfg.get('batch_size') * self.cfg.get('grid_height')]),
                        self.cfg.get('boxes'), axis=0),
                output_shape))

        cell_y = tf.transpose(cell_x, (0, 2, 1, 3))

        xpred = y_pred[:, :, :, :, 1]
        ypred = y_pred[:, :, :, :, 2]
        wpred = y_pred[:, :, :, :, 3]
        hpred = y_pred[:, :, :, :, 4]

        box_xpred = (tf.sigmoid(xpred) + cell_x) * self.cfg.get('cell_width')
        box_ypred = (tf.sigmoid(ypred) + cell_y) * self.cfg.get('cell_height')
        box_wpred = tf.exp(wpred) * self.cfg.get('anchors')[:, 0] * self.cfg.get('cell_width')
        box_hpred = tf.exp(hpred) * self.cfg.get('anchors')[:, 1] * self.cfg.get('cell_height')

        box_wpredhalf = box_wpred / 2
        box_hpredhalf = box_hpred / 2

        box_xpredmin = box_xpred - box_wpredhalf
        box_xpredmax = box_xpred + box_wpredhalf
        box_ypredmin = box_ypred - box_hpredhalf
        box_ypredmax = box_ypred + box_hpredhalf

        box_xtrue = y_true[:, :, :, :, 1]
        box_ytrue = y_true[:, :, :, :, 2]
        box_wtrue = y_true[:, :, :, :, 3]
        box_htrue = y_true[:, :, :, :, 4]

        box_wtruehalf = box_wtrue / 2
        box_htruehalf = box_htrue / 2

        box_xtruemin = box_xtrue - box_wtruehalf
        box_xtruemax = box_xtrue + box_wtruehalf
        box_ytruemin = box_ytrue - box_htruehalf
        box_ytruemax = box_ytrue + box_htruehalf

        interxmins = tf.maximum(box_xpredmin, box_xtruemin)
        interymins = tf.maximum(box_ypredmin, box_ytruemin)
        interxmaxes = tf.minimum(box_xpredmax, box_xtruemax)
        interymaxes = tf.minimum(box_ypredmax, box_ytruemax)

        interareas = tf.maximum(0.0, interxmaxes - interxmins + 1) * tf.maximum(0.0, interymaxes - interymins + 1)

        trueareas = (box_htrue + 1) * (box_wtrue + 1)
        predareas = (box_hpred + 1) * (box_wpred + 1)

        ious = interareas / (trueareas + predareas - interareas)

        mask_shape = output_shape

        objs = tf.ones(shape=(mask_shape)) * self.cfg.get('object_scale')
        noobjs = tf.ones(shape=(mask_shape)) * self.cfg.get('noobject_scale')
        coords = tf.ones(shape=(mask_shape)) * self.cfg.get('coord_scale')
        classes = tf.ones(shape=(mask_shape)) * self.cfg.get('class_scale')
        zeros = tf.zeros(shape=(mask_shape))

        objects_present = tf.greater(c_true, 0.0)
        confcoef = tf.where(objects_present, objs, noobjs)
        coordcoef = tf.where(objects_present, coords, zeros)
        classescoef = tf.where(objects_present, classes, zeros)

        xtrue = box_xtrue / self.cfg.get('cell_width') - (cell_x + 0.5)
        ytrue = box_ytrue / self.cfg.get('cell_height') - (cell_y + 0.5)
        wtrue = tf.log(box_wtrue / (self.cfg.get('cell_width') * self.cfg.get('anchors')[:, 0]))
        htrue = tf.log(box_htrue / (self.cfg.get('cell_height') * self.cfg.get('anchors')[:, 1]))

        wtrue = tf.where(objects_present, wtrue, zeros)
        htrue = tf.where(objects_present, htrue, zeros)
        ious = tf.where(objects_present, ious, zeros)

        classestrue = tf.argmax(y_true[:, :, :, :, 5:], -1)
        classespred = tf.nn.softmax(y_pred[:, :, :, :, 5:])

        classesloss = classescoef * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classestrue, logits=classespred)

        confloss = confcoef * ((ious - c_pred) ** 2)

        xloss = coordcoef * ((xtrue - xpred) ** 2)
        yloss = coordcoef * ((ytrue - ypred) ** 2)

        wloss = coordcoef * ((wtrue - wpred) ** 2)
        hloss = coordcoef * ((htrue - hpred) ** 2)

        coordloss = xloss + yloss + wloss + hloss

        loss = confloss + coordloss + classesloss

        return loss

def main():
    labels_dir = "./labels.txt"
    cfg_path = r"C:\Users\Gencho\Desktop\ObjectDetection\experiments\mobilenetyolov2-voc.cfg"

    cfg = Config(cfg_path)
    encoder = LabelEncoder(labels_dir)

main()