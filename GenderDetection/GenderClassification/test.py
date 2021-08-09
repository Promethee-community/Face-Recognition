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
from keras.models import load_model
import scipy.misc
from scipy.special import comb
import tensorflow as tf

# we are only going to use 4 attributes
#sklearn.neural_network.multilayer_perceptron()
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
Flagsnap=True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import face_recognition


def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    #print(X_img)
    #print(X_img.shape)
    #print(img_path)
    #print(X_img.shape[0])
    #print(X_img.shape[1])
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    #print(locs)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs

def predict_one_image(img_path, model_path):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        print("No face in "+img_path)
        return None, None
    predr=[]
    classifier = load_model(os.path.realpath(model_path))
    for faces_encoding in face_encodings :
    #    print("shape",faces_encoding.reshape(1,128).shape)
        result = classifier.predict(faces_encoding.reshape(1,128))
    #    print("prediction result:",result)
    return result, locs
def draw_attributes(img_path, df):
    """Write bounding boxes and predicted face attributes on the image
    """
    img = cv2.imread(img_path)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
            
        else:
            gender = 'Female'

       

        race = np.argmax(row[1][1:4])
        text_showed = "{} {}".format(race, gender)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = img.shape[1]
        cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    return img



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, 
                        default='../Dataset/mypicture_data', 
                        help='input image directory (default: mypicture_data)')
    parser.add_argument('--output_dir', type=str, 
                        default='mypicture_results/', 
                        help='output directory to save the results (default: mypicture_results/)')
    parser.add_argument('--model', type=str, required = False,
                        default='train/GenderDetection_v1-0.0533-98.27.h5', 
                        help='path to trained model (default: our model)')

    args = parser.parse_args()
    output_dir = args.output_dir
    input_dir = args.img_dir
    model_path = args.model

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # load the model
   
    classifier=None

    print("classifying images in {}".format(input_dir))
    for fname in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, fname)
        #print(img_path)
        if img_path[-3:]=="jpg" :
            #try:
            print("predict_one_image",img_path)
            pred, locs = predict_one_image(img_path, model_path)
           # except:
           #     print("Skipping {}".format(img_path))
            if not locs:
                continue

            def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
                dim = None
                (h, w) = image.shape[:2]

                if width is None and height is None:
                    return image
                if width is None:
                    r = height / float(h)
                    dim = (int(w * r), height)
                else:
                    r = width / float(w)
                    dim = (width, int(h * r))

                return cv2.resize(image, dim, interpolation=inter)
            
            myimg = cv2.imread(img_path)
            imgresized = ResizeWithAspectRatio(myimg, width=500)
            if pred[0]>0.5:
                cv2.imshow('It\'s a Man !', imgresized)
            else:
                cv2.imshow('It\'s a Woman !', imgresized)

            cv2.waitKey() #displays the last picture on the added on the folder ... so let's decide if we treat only 1 picture ? 

            cv2.destroyAllWindows()
         

if __name__ == "__main__":
    main()
