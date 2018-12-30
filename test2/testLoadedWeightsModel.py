import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
import keras.losses
import os



imgDir = "images"
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

recognitionFile = open("loadedWeightsModel.txt", 'w')
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
print("loading model")


FRmodel = load_model("FaceRecoModelWithWeights.h5", custom_objects= {'triplet_loss': triplet_loss})
print("Model finished loading")



def prepare_database():
    database = {}

    for root, dirs, files in os.walk(imgDir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                idolNameDirPath = os.path.dirname(path)
                identity = os.path.basename(idolNameDirPath)                
                print("Training: " + identity + " --path: " + path)

                #center the face.
                face = getFace(path)

                #TODO: we might want to accumulate the encodings of multiple pictures
                if not identity in database:                
                    database[identity] = [img_to_encoding(face, FRmodel)]
                else:
                    database[identity].append(img_to_encoding(face, FRmodel))

    #iterate through each value (list) of each key and calculate the average encoding value.
    for (id, encList) in database.items():
        meanEnc = encList[0]
        if len(encList) <= 1:
            database[id] = meanEnc
            continue
        
        for  i in range (1, len(encList)):
            meanEnc += encList[i]
        meanEnc = meanEnc / len(encList)
        
        database[id] = meanEnc
    
    return database

def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None

    recognitionFile.write("Recognizing a face:--\n")
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        recognitionFile.write('distance for %s is %s\n' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return None
    else:
        return identity

def getFace(imgPath):
    roiImg = None
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    faces = faceDetector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(str(len(faces)) + " --path: "  + imgPath)
    x, y, w, h = getLargestFace(faces, img)

    
    roiGray = gray[y:y+h, x:x+w]
    grayCopy = gray.copy()
    cv2.rectangle(grayCopy, (x, y), (x+w, y+h), (255,0,0), 1)
    cv2.imshow("get face of prediction", grayCopy)
    cv2.waitKey(0)        
    roiImg = img[y:y+h, x:x+w]
    return roiImg


def getLargestFace(faces, img):
    largestArea = 0
    largestFace = None
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        height, width = roi.shape[:2]
        area = height * width
        if area > largestArea:
            largestArea = area
            largestFace = (x, y, w, h)
    return largestFace

        
    
testImgDir = "testAgainstImages/"
print("Preparing database--")
db = prepare_database()
print("Database prepared--")


for i in range (7):
    filePath = testImgDir + str(i+1) + ".jpg"
    face = getFace(filePath)
    print("--Recognizing a face:")
    print(who_is_it(face, db, FRmodel))
    
'''
for i in range (2):
    filePath = testImgDir + str(i+5) + ".png"
    face = getFace(filePath)
    print("--Recognizing a face:")
    print(who_is_it(face, db, FRmodel))
          


'''
