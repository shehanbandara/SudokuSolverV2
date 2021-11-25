import cv2
import numpy as np
import sudokuSolver
from tensorflow.keras.models import load_model

# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load the digit recognition CNN Model
model = load_model('digitRecognitionCNNModel.h5')

while(True):
    ret, frame = cap.read()
    if ret == True:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
