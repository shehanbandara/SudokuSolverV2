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

    # Read from the webcam
    ret, frame = cap.read()

    # If the frame is available
    if ret == True:

        #
        cv2.imshow('Sudoku Solver V2', frame)

        # Press 'q' to stop the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the frame is unavailable stop the program
    else:
        break

# Release & destroy all resources
cap.release()
cv2.destroyAllWindows()
