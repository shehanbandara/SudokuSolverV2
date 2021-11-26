import cv2
import matplotlib.pyplot as plt
import numpy as np
import sudokuSolver


def imagePreProcessing(image):

    # Blur the image
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def findBestContour(image):
    # Apply an adaptive threshold to the image
    threshold = cv2.adaptiveThreshold(image, 255, 0, 1, 19, 2)

    # Find the contours
    contours, hierarchy = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize maximum area and best contour variables
    maxArea = 0
    bestContour = None

    # Loop through all contours
    for i in contours:
        area = cv2.contourArea(i)

        # Take a closer look if possibly the maximum area
        if area > 1000:

            # If maximum area
            if area > maxArea:

                # Overwrite maximum area and best contour
                maxArea = area
                bestContour = i

    return bestContour


def solve(frame, model):

    # Make a copy of the Sudoku Puzzle to be used later
    originalCopy = np.copy(frame)

    # Preform some pre-processing to the image
    image = imagePreProcessing(frame)

    # Find the biggest contour in the image (the Sudoku Puzzle)
    bestContour = findBestContour(image)

    # Return the original image if there is no Sudoku Puzzle yet
    if bestContour is None:
        return originalCopy
