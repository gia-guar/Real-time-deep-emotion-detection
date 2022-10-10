## IMPORT DEPENDANCIES
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import math

## UPLOAD THE MODELS
models = []
models.append( load_model(os.path.join('models','from_sad_to_happy.h5')) )
models.append(load_model(os.path.join('models','from_disgust_to_happy.h5')) )
models.append( load_model(os.path.join('models','from_surprise_to_happy.h5')) )
models.append( load_model(os.path.join('models','from_neutral_to_happy.h5')) )
models.append( load_model(os.path.join('models','from_angry_to_happy.h5')) )


## INITIALIZE THE CAMERA
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

## retrieve the input shape of the net
input_shape=(256,256,3)

# Evaluation function
def generate_opinions(models,frame,rule='majority'):
    predictions = []
    for model in models:
        y_pred = model.predict(frame)
        predictions.append(y_pred)
    
    if rule == 'majority':
        predictions = np.around(np.array(predictions))
        confidence = math.fsum(predictions)/(len(predictions))
        
        if confidence>=0.5:
            return (confidence,'happy')
        else:
            return (confidence, 'not happy')
        
    if rule == 'average':
        confidence = math.fsum(predictions)/(len(predictions))
        if confidence>0.5:
            return (confidence, 'happy' )
        return (confidence, 'not happy')

while cap.isOpened(): 
    # capture frames (CV images)
    ret, frame = cap.read()

    # convert images in inputs
    resize = tf.image.resize(frame, (input_shape[0], input_shape[1]) )
    image = np.expand_dims(resize/255, 0)

    # feed the nets
    confidence, label = generate_opinions(models,image, rule='average')
    confidence = np.around(confidence, 2)
    
    # Display prediction on screen
    x,y,w,h = 0,0,400,250
    cv2.putText(img=frame, text=label,org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255*(1-confidence),0,255*confidence), thickness=5)
    x,y,w,h = 75,40,100,100
    cv2.putText(img=frame, text=str(confidence),org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255*(1-confidence),0,255*confidence), thickness=5)

    cv2.imshow('prediction',frame)
    cv2.waitKey(1)

    # exit option by pressing 'q' (CAPS LOCK sensitive!)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

## CLOSE THE CAMERA
cap.release()