import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import cv2
import numpy as np
from keras import backend as K
import os


imgDir = "images"
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#our loss function passed as arg during model compile.
def triplet_loss(yTrue, yPred, alpha=0.3):
    anchor, positive, negative = yPred[0], yPred[1], yPred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, 
               negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
   
    return loss

#we need these globals on top so that other functions can work . TODO: Refactor later.
K.set_image_data_format('channels_first')

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

FRmodel.compile(optimizer = "adam", loss = triplet_loss, metrics = ['accuracy'])
print("loading weights from FaceNet")
load_weights_from_FaceNet(FRmodel)

FRmodel.save("FaceRecoModelWithWeights.h5")
