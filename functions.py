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
        if area > 40000:

            # If maximum area
            if area > maxArea:

                # Overwrite maximum area and best contour
                maxArea = area
                bestContour = i

    return bestContour


def fourCornersOfBestContour(bestContour):

    # Initialize variables
    iterations = 0
    coefficient = 1

    # Continue looping untill 200 iterations or a coefficient < 0
    while iterations < 200 and coefficient >= 0:

        # Increment the iteration counter
        iterations += 1

        # Calculate an accuracy paramter
        epsilon = coefficient * cv2.arcLength(bestContour, True)

        # Approximate the shape of the best contour
        approximation = cv2.approxPolyDP(bestContour, epsilon, True)

        # Find the convex hull of the best contour
        hull = cv2.convexHull(approximation)

        # If the convex hull has 4 edges (4 corners)
        if len(hull) == 4:

            # Return the 4 corners
            return hull

       # If the convex hull does not have 4 edges (4 corners)
        else:

            # Increment the coefficient if more than 4 edges
            if len(hull) > 4:
                coefficient += .01

            # Decrement the coefficient if less than 4 edges
            else:
                coefficient -= .01

    # Return None if the best contour does not have 4 corners
    return None


def solve(frame, model):

    # Make a copy of the Sudoku Puzzle to be used later
    originalCopy = np.copy(frame)

    # Preform some pre-processing to the image
    image = imagePreProcessing(frame)

    # Find the best contour in the image (the Sudoku Puzzle)
    bestContour = findBestContour(image)

    # Return the original image if there is no Sudoku Puzzle yet
    if bestContour is None:
        return originalCopy

    # Find the 4 corners of the best contour
    fourCorners = fourCornersOfBestContour(bestContour)

    # Return the original image if there is no Sudoku Puzzle yet
    if fourCorners is None:
        return originalCopy
