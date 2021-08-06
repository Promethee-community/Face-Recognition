from __future__ import print_function
import argparse
import os

import numpy as np
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
import numpy
from keras.models import load_model


# we are only going to use 4 attributes
#sklearn.neural_network.multilayer_perceptron()
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
Flagsnap=True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


X = numpy.load("X_val_f.npy")
Y = numpy.load("Y_val_f.npy")

classifier = load_model(os.path.realpath('train/GenderDetection_v1-0.054-98.27.h5'))
print(classifier.evaluate(X,Y))

result=classifier.predict(X)

error=0
for i in range(0,len(result)):
    if(result[i]<0.5 and Y[i]>=0.5):
        error+=1
    if(result[i]>=0.5 and Y[i]<0.5):
        error+=1    
print("nb errore",error)
print("% error",100.0-((error/len(result))*100.0))

