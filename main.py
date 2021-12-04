import cv2
import functions
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import numpy as np


# Function to display the Sudoku Puzzle solution
def show(image, width, height):
    image = cv2.resize(image, (width, height))
    cv2.imshow('Sudoku Solver V2', image)


# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load the digit recognition Model
# Load the weights and configuration seperately to speed up the prediction
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))
model.load_weights("digitRecognitionModel.h5")

while(True):

    # Read from the webcam
    ret, frame = cap.read()

    # If the frame is available
    if ret == True:

        # Solve the Sudoku Puzzle and overlay the solution
        sudokuSolution = functions.solve(frame, model)

        # Display the Sudoku Puzzle solution
        show(sudokuSolution, 1066, 600)

        # Press 'q' to stop the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # If the frame is unavailable stop the program
    else:
        break

# Release & destroy all resources
cap.release()
cv2.destroyAllWindows()
