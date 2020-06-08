import random

import numpy as np
import tensorflow as tf
from PIL import Image as Img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from nms import nms, group_nms
from sklearn.model_selection import train_test_split
from utils import parse_annotation, Object, calculate_IoU, draw_image, save_objects_to_json
from utils import softmax, sigmoid

from ObjectDetectionElsys.augmentation import read_image


class YOLO():
    def __init__(self, cfg, encoder, networkfactory, weights=None, optimizer=None):
        self.cfg = cfg
        self.encoder = encoder
        self.networkfactory = networkfactory
        self.model = networkfactory.get_network(cfg, optimizer, self.custom_loss, weights)

    def encode_y_true_from_annotation(self, annotation, raw_file=True):
        y_true = np.zeros(shape=(self.cfg.get('grid_width'), self.cfg.get('grid_height'),
                                 self.cfg.get('boxes'), 5 + self.cfg.get('classes')))
        objs = [[[] for col in range(self.cfg.get('grid_width'))] for row in range(self.cfg.get('grid_height'))]

        if raw_file:
            annotation = parse_annotation(annotation)

        image_cell_width = annotation.imagewidth / self.cfg.get('grid_width')
        image_cell_height = annotation.imageheight / self.cfg.get('grid_height')

        for obj in annotation.objects:
            if not self.encoder.supports(obj.name):
                continue

            obj.xmid = (obj.xmax + obj.xmin) / 2
            obj.ymid = (obj.ymax + obj.ymin) / 2
            obj.width = obj.xmax - obj.xmin
            obj.height = obj.ymax - obj.ymin

            row = int(obj.ymid / image_cell_height)
            col = int(obj.xmid / image_cell_width)

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

        output_shape = (
            self.cfg.get('batch_size'), self.cfg.get('grid_width'), self.cfg.get('grid_height'), self.cfg.get('boxes'))

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

        box_wpred = (tf.sigmoid(wpred) + 0.5) * self.cfg.get('anchors')[:, 0] * self.cfg.get('cell_width')
        box_hpred = (tf.sigmoid(hpred) + 0.5) * self.cfg.get('anchors')[:, 1] * self.cfg.get('cell_height')

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

        ious = tf.where(objects_present, ious, zeros)

        classestrue = tf.argmax(y_true[:, :, :, :, 5:], -1)
        classespred = tf.nn.softmax(y_pred[:, :, :, :, 5:])

        classesloss = classescoef * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classestrue,
                                                                                   logits=classespred)

        confloss = confcoef * ((ious - c_pred) ** 2)

        xloss = coordcoef * ((box_xtrue - box_xpred) ** 2)
        yloss = coordcoef * ((box_ytrue - box_ypred) ** 2)
        wloss = coordcoef * ((box_wtrue - box_wpred) ** 2)
        hloss = coordcoef * ((box_htrue - box_hpred) ** 2)

        coordloss = xloss + yloss + wloss + hloss

        loss = confloss + coordloss + classesloss

        return loss

    def decode_prediction(self, y_pred, onlyconf=False):
        objects = []

        for row in range(self.cfg.get('grid_height')):
            for col in range(self.cfg.get('grid_width')):
                for box in range(self.cfg.get('boxes')):
                    to, tx, ty, tw, th = y_pred[row, col, box, :5]

                    conf = sigmoid(to)
                    labels = y_pred[row, col, box, 5:]
                    labels = softmax(labels)

                    max_label = max(labels)

                    conf *= max_label

                    if conf >= self.cfg.get('threshhold'):
                        max_index = -1
                        for i in range(len(labels)):
                            if labels[i] == max_label:
                                max_index = i
                                break

                        label = self.encoder.decode(max_index)
                        if onlyconf:
                            bx = (col + 0.5) * self.cfg.get('cell_width')
                            by = (row + 0.5) * self.cfg.get('cell_height')

                            pw, ph = self.cfg.get('anchors')[box]

                            bw = pw * self.cfg.get('cell_width')
                            bh = ph * self.cfg.get('cell_height')
                        else:
                            bx = (sigmoid(tx) + col) * self.cfg.get('cell_width')
                            by = (sigmoid(ty) + row) * self.cfg.get('cell_height')

                            pw, ph = self.cfg.get('anchors')[box]

                            bw = pw * (sigmoid(tw) + 0.5)
                            bh = ph * (sigmoid(th) + 0.5)

                            # bw = pw * np.exp(tw)
                            # bh = ph * np.exp(th)

                            bw *= self.cfg.get('cell_width')
                            bh *= self.cfg.get('cell_height')

                        objects.append(
                            Object(xmin=bx - bw / 2, xmax=bx + bw / 2, ymin=by - bh / 2, ymax=by + bh / 2, conf=conf,
                                   name=label))

        return objects

    def feed_forward_batch(self, images, supression="none", onlyconf=False):
        ins = []

        scales = []
        normalizer = self.networkfactory.get_normalizer(self.cfg)
        for image in images:
            width_scale = image.shape[0] / self.cfg.get('image_width')
            height_scale = image.shape[1] / self.cfg.get('image_height')

            scales.append((width_scale, height_scale))

            ins.append(normalizer(image))

        y_preds = self.model.predict(np.array(ins, dtype=np.float32))

        result = []
        for pred in y_preds:
            objects = self.decode_prediction(pred, onlyconf)

            if supression == "group":
                objects = group_nms(self.cfg, objects)
            if supression == "regular":
                objects = nms(self.cfg, objects)

            result.append(objects)

        return result

    def feed_forward(self, image_path, draw=False, supression="none", save_image=False, save_json=False,
                     onlyconf=False):
        im = Img.open(image_path)

        width_scale = im.width / self.cfg.get('image_width')
        height_scale = im.height / self.cfg.get('image_height')

        im = read_image(image_path, inputshape=(self.cfg.get('image_width'), self.cfg.get('image_height')))
        im = self.networkfactory.get_normalizer(self.cfg)(im)

        y_pred = self.model.predict(np.array([im]))[0]

        objects = self.decode_prediction(y_pred, onlyconf)

        for obj in objects:
            obj.xmin *= width_scale
            obj.xmax *= width_scale
            obj.ymin *= height_scale
            obj.ymax *= height_scale

        if supression == "group":
            objects = group_nms(self.cfg, objects)
        if supression == "regular":
            objects = nms(self.cfg, objects)

        if draw or save_image:
            draw_image(image_path, objects, draw_grid=False,
                       grid_size=(self.cfg.get('grid_width'), self.cfg.get('grid_height')), save=save_image, displ=draw)

        if save_json:
            save_objects_to_json(image_path, objects)

        return objects

    def train(self, generator, annotations, images, epochs, checkpoint_period=None, early_stopping=False, patience=20,
              validation_portion=None):

        if validation_portion == None:
            validation_portion = self.cfg.get('batch_size') / len(images)
            if validation_portion > 0.2:
                validation_portion = 0.2

        annotations_train, annotations_val, images_train, images_val = train_test_split(annotations, images,
                                                                                        test_size=validation_portion)

        gen_train = generator(annotations_train, images_train, self.cfg,
                              self.networkfactory.get_normalizer(self.cfg),
                              self.encode_y_true_from_annotation)

        gen_val = generator(annotations_val, images_val, self.cfg,
                            self.networkfactory.get_normalizer(self.cfg),
                            self.encode_y_true_from_annotation)

        callbacks = []

        if checkpoint_period:
            modelname = self.cfg.get("net")
            filepath = "./weights/" + modelname + "yolov2-checkpoint-{epoch:02d}"

            checkpoint = ModelCheckpoint(filepath=filepath, period=checkpoint_period)
            callbacks.append(checkpoint)

        if early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
            callbacks.append(es)

        steps_per_epoch = len(images_train) // self.cfg.get('batch_size')
        validation_steps = len(images_val) // self.cfg.get('batch_size')

        if validation_steps == 0:
            validation_steps = 1

        h = self.model.fit_generator(gen_train, steps_per_epoch=steps_per_epoch,
                                     validation_data=gen_val,
                                     validation_steps=validation_steps,
                                     epochs=epochs,
                                     callbacks=callbacks)

        return h

    def save(self, path):
        self.model.save(path)
