import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

def maxPool():
    return MaxPooling2D(pool_size=(2, 2),border_mode='valid')

def getConv1024():
    return Convolution2D(1024,3,3 ,border_mode='same')

def getNN():
    model = Sequential()
    model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(maxPool())
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(maxPool())
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(maxPool())
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(maxPool())
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(maxPool())
    model.add(getConv1024())
    model.add(LeakyReLU(alpha=0.1))
    model.add(getConv1024())
    model.add(LeakyReLU(alpha=0.1))
    model.add(getConv1024())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model

class Box:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.w = 0.
        self.h = 0.
        self.c = 0.
        self.prob = 0.

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection_area(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area

def box_union(a, b):
    i = box_intersection_area(a, b)
    return a.w * a.h + b.w * b.h - i

def box_iou(a, b):
    return box_intersection_area(a, b) / box_union(a, b)

def get_boxs(net_out, threshold = 0.2, sqrt=1.8,C=20, B=2, S=7):

    class_num = 6 # class car
    boxes = []
    SS        =  S * S
    prob_size = SS * C
    conf_size = SS * B

    probs = net_out[0 : prob_size]
    confs = net_out[prob_size : (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size) : ]

    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])

    for grid in range(SS):
        for b in range(B):
            bx   = Box()
            bx.c =  confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid %  S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w =  cords[grid, b, 2] ** sqrt
            bx.h =  cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c

            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                boxes.append(bx)
    return boxes

def yolo_boxes(net_out, threshold = 0.2, sqrt=1.8,C=20, B=2, S=7):
    boxes = get_boxs(net_out,threshold,sqrt,C,B,S)

    # combine boxes that are overlap
    boxes.sort(key=lambda b:b.prob,reverse=True)
    for i in range(len(boxes)):
        boxi = boxes[i]
        if boxi.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            boxj = boxes[j]
            if box_iou(boxi, boxj) >= .4:
                boxes[j].prob = 0.
    boxes = [b for b in boxes if b.prob > 0.]

    return boxes

def draw_box(boxes,im,crop_dim):
    cp_img = im.copy()
    [xmin, xmax] = crop_dim[0]
    [ymin, ymax] = crop_dim[1]
    h, w, _ = cp_img.shape

    for b in boxes:
        w = xmax - xmin
        h = ymax - ymin

        x1  = int ((b.x - b.w/2.) * w) + xmin
        x2 = int ((b.x + b.w/2.) * w) + xmin
        y1   = int ((b.y - b.h/2.) * h) + ymin
        y1   = int ((b.y + b.h/2.) * h) + ymin

        if x1  < 0:
            x1 = 0
        if x2 > w - 1:
            x2 = w - 1
        if y1 < 0:
            y1 = 0
        if y2>h - 1:
            y2 = h - 1

        cv2.rectangle(cp_img, (x1, y1), (x2, y2), (255,0,0), 3)

    return cp_img
