import numpy as np 
import cv2

from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

model = load_model('test.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNo): 
    if classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'Speed Limit 100 km/h'
    elif classNo == 7: return 'Speed Limit 120 km/h'
    else: return 'Unknown'

while True: 
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32,32))
    img = preprocessing(img)
    img = img.reshape(1,32,32,1)

    predictions = model.predict(img)
    indexVal = np.argmax(predictions)
    probability = np.max(predictions)
    print(indexVal, probability)

    cv2.putText(imgOriginal, str(getClassName(indexVal)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

    cv2.putText(imgOriginal, str(round(probability * 100, 2)) + '%', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)

    cv2.imshow('Result', imgOriginal)
    cv2.waitKey(1)
