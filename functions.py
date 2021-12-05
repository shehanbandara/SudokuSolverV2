import copy
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import sudokuSolver
from scipy import ndimage

# Gloabl iteration counter variable
iterations = 0


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


def locateCorners(fourCorners):

    # Initialize an array to hold the coordinates of the 4 corners
    arrayOfCorners = np.zeros((4, 2), dtype="float32")

    # Reshape the 4 corners
    fourCorners = fourCorners.reshape(4, 2)

    # Initialize a variable to store the sum of the coordinates
    sumOfCoordinates = 9999999

    # Initialize a variable to store the index of the coordinate of interest
    index = 0

    # Find the top left corner (sum of the coordinates is the smallest)
    for i in range(4):
        if(fourCorners[i][0] + fourCorners[i][1] < sumOfCoordinates):
            sumOfCoordinates = fourCorners[i][0] + fourCorners[i][1]
            index = i

    # Store the top left corner
    arrayOfCorners[0] = fourCorners[index]

    # Delete the top left corner from the 4 corners
    threeCorners = np.delete(fourCorners, index, 0)

    # Reset variable to store the sum of the coordinates
    sumOfCoordinates = 0

    # Find the bottom right corner (sum of the coordinates is the largest)
    for i in range(3):
        if(threeCorners[i][0] + threeCorners[i][1] > sumOfCoordinates):
            sumOfCoordinates = threeCorners[i][0] + threeCorners[i][1]
            index = i

    # Store the bottom right corner
    arrayOfCorners[3] = threeCorners[index]

    # Delete the top left corner from the 4 corners
    twoCorners = np.delete(threeCorners, index, 0)

    # Find the top right & bottom left corners
    if (twoCorners[0][0] > twoCorners[1][0]):
        arrayOfCorners[1] = twoCorners[0]
        arrayOfCorners[2] = twoCorners[1]
    else:
        arrayOfCorners[1] = twoCorners[1]
        arrayOfCorners[2] = twoCorners[0]

    # Ensure shape
    arrayOfCorners = arrayOfCorners.reshape(4, 2)

    return arrayOfCorners


def square(corners):

    # Function to find the angle between 2 vectors
    def angleBetweenVectors(vector1, vector2):
        unitVector1 = vector1 / np.linalg.norm(vector1)
        unitVector2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unitVector1, unitVector2)
        angle = np.arccos(dot_product) * 57.2958
        return angle

    TL = corners[0]
    TR = corners[1]
    BL = corners[2]
    BR = corners[3]

    # TL - TR
    # |     |
    # BL - BR

    # Vectors between the 4 corners
    TLTR = TR - TL
    TRBR = BR - TR
    BRBL = BL - BR
    BLTL = TL - BL

    # Angles at the 4 corners
    angleTL = angleBetweenVectors(BLTL, TLTR)
    angleTR = angleBetweenVectors(TLTR, TRBR)
    angleBR = angleBetweenVectors(TRBR, BRBL)
    angleBL = angleBetweenVectors(BRBL, BLTL)

    # Are the angles at the 4 corners approximately 90 degrees
    rightAngleTL = (angleTL - 90) < 20
    rightAngleTR = (angleTR - 90) < 20
    rightAngleBR = (angleBR - 90) < 20
    rightAngleBL = (angleBL - 90) < 20

    # If all 4 corners are not approximately 90 degrees return False
    if not ((rightAngleTL and rightAngleTR) and (rightAngleBL and rightAngleBR)):
        return False

    # Lengths of the 4 sides
    lengthTLTR = math.sqrt(((TL[0] - TR[0]) ** 2) + ((TL[1] - TR[1]) ** 2))
    lengthTRBR = math.sqrt(((TR[0] - BR[0]) ** 2) + ((TR[1] - BR[1]) ** 2))
    lengthBRBL = math.sqrt(((BR[0] - BL[0]) ** 2) + ((BR[1] - BL[1]) ** 2))
    lengthBLTL = math.sqrt(((BL[0] - TL[0]) ** 2) + ((BL[1] - TL[1]) ** 2))

    # Shortest & longest side lengths
    shortestSide = min(lengthTLTR, lengthTRBR, lengthBRBL, lengthBLTL)
    longestSide = max(lengthTLTR, lengthTRBR, lengthBRBL, lengthBLTL)

    # If all 4 sides are not approximately the same length return False
    if (longestSide > shortestSide * 1.2):
        return False

    # Return True if the Sudoku Puzzle is a square
    return True


def grabSudokuPuzzleBoard(corners, image):

    TL = corners[0]
    TR = corners[1]
    BL = corners[2]
    BR = corners[3]

    # TL - TR
    # |     |
    # BL - BR

    # Lengths of the 4 sides
    lengthTLTR = math.sqrt(((TL[0] - TR[0]) ** 2) + ((TL[1] - TR[1]) ** 2))
    lengthTRBR = math.sqrt(((TR[0] - BR[0]) ** 2) + ((TR[1] - BR[1]) ** 2))
    lengthBRBL = math.sqrt(((BR[0] - BL[0]) ** 2) + ((BR[1] - BL[1]) ** 2))
    lengthBLTL = math.sqrt(((BL[0] - TL[0]) ** 2) + ((BL[1] - TL[1]) ** 2))

    # Height and width of the Sudoku Puzzle
    height = max(int(lengthBLTL), int(lengthTRBR))
    width = max(int(lengthTLTR), int(lengthBRBL))

    # Construct Sudoku Puzzle board destination array
    destinationArray = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype="float32")

    # Perform the perspective transform
    value = cv2.getPerspectiveTransform(corners, destinationArray)

    # Warp the Sudoku Puzzle board
    sudokuPuzzleBoard = cv2.warpPerspective(image, value, (width, height))

    return sudokuPuzzleBoard


def processSudokuPuzzleBoard(sudokuPuzzleBoard):

    # Blur the Sudoku Puzzle board
    image = cv2.GaussianBlur(sudokuPuzzleBoard, (5, 5), 0)

    # Convert Sudoku Puzzle board to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply an adaptive threshold to the Sudoku Puzzle board
    image = cv2.adaptiveThreshold(image, 255, 1, 1, 11, 2)

    # Invert the color of the Sudoku Puzzle board
    image = cv2.bitwise_not(image)

    # Apply a threshold to the Sudoku Puzzle board
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    return image


def extractAndClassifyDigits(processedSudokuPuzzleBoard, model):

    def seperateDigitAndRemoveNoise(box):

        # Convert the image from type float to type integer
        box = box.astype('uint8')

        # Calculate the connected components of the image
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            box, connectivity=8)
        size = stats[:, -1]

        # If there is 1 or less connected components (no digit) return a blank white image
        if (len(size) <= 1):
            blankBox = np.zeros(box.shape)
            blankBox.fill(255)
            return blankBox

        # Initialize variables for the biggest label and the biggest label's size
        biggestLabel = 1
        biggestLabelSize = size[1]

        for i in range(2, retval):
            if size[i] > biggestLabelSize:
                biggestLabel = i
                biggestLabelSize = size[i]

        # Remove all noise from the digit
        digitBox = np.zeros(box.shape)
        digitBox.fill(255)
        digitBox[labels == biggestLabel] = 0

        return digitBox

    # Initialize a list to store the Sudoku Puzzle digits
    digits = []
    for i in range(9):
        row = []
        for j in range(9):
            row.append(0)
        digits.append(row)

    # Find the height and width of each box
    boxHeight = processedSudokuPuzzleBoard.shape[0] // 9
    boxWidth = processedSudokuPuzzleBoard.shape[1] // 9

    # Calculate a height and width to crop each box by to remove the boundary lines
    boxCroppedHeight = math.floor(boxHeight / 10)
    boxCroppedWidth = math.floor(boxWidth / 10)

    # Loop through each box
    for i in range(9):
        for j in range(9):

            # Crop each box to remove the boundary lines
            croppedBox = processedSudokuPuzzleBoard[boxHeight*i+boxCroppedHeight:boxHeight*(
                i+1)-boxCroppedHeight, boxWidth*j+boxCroppedWidth:boxWidth*(j+1)-boxCroppedWidth]

            # Remove all black lines near the edges
            # Top side
            while np.sum(croppedBox[0]) <= (1 - 0.6) * croppedBox.shape[1] * 255:
                croppedBox = croppedBox[1:]
            # Bottom side
            while np.sum(croppedBox[:, -1]) <= (1 - 0.6) * croppedBox.shape[1] * 255:
                croppedBox = np.delete(croppedBox, -1, 1)
            # Left side
            while np.sum(croppedBox[:, 0]) <= (1 - 0.6) * croppedBox.shape[0] * 255:
                croppedBox = np.delete(croppedBox, 0, 1)
            # Right side
            while np.sum(croppedBox[-1]) <= (1 - 0.6) * croppedBox.shape[0] * 255:
                croppedBox = croppedBox[:-1]

            # Seperate the digit in each box and remove any noise
            digitBox = cv2.bitwise_not(croppedBox)
            digitBox = seperateDigitAndRemoveNoise(digitBox)

            # Resize the box
            digitBox = cv2.resize(digitBox, (28, 28))

            # If the box has very little black pixels it is blank so overwrite with 0
            if digitBox.sum() >= ((28**2) * 255) - (28 * 255):
                digits[i][j] = 0
                continue

            # If the box has a large white area in its center it is blank so overwrite with 0
            boxCenterHeight = digitBox.shape[0] // 2
            boxCenterWidth = digitBox.shape[1] // 2
            boxCenter = digitBox[boxCenterHeight//2:boxCenterHeight//2 +
                                 boxCenterHeight, boxCenterWidth//2:boxCenterWidth//2+boxCenterWidth]
            if boxCenter.sum() >= boxCenterHeight * boxCenterWidth * 255 - 255:
                digits[i][j] = 0
                continue

            # Apply a Binary Threshold to make the digit clearer
            _, digitBox = cv2.threshold(digitBox, 200, 255, cv2.THRESH_BINARY)
            digitBox = digitBox.astype(np.uint8)

            # Invert the color of the box
            digitBox = cv2.bitwise_not(digitBox)

            # Store the rows and columns
            rows, columns = digitBox.shape

            # Calculate the center of mass of the box
            centerY, CenterX = ndimage.measurements.center_of_mass(digitBox)

            # Calculate what to shift the box by to center it
            shiftX = np.round(columns / 2.0 - CenterX).astype(int)
            shiftY = np.round(rows / 2.0 - centerY).astype(int)

            # Shift the box
            temp = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
            shiftedDigitBox = cv2.warpAffine(digitBox, temp, (columns, rows))

            # Invert the color of the box
            shiftedDigitBox = cv2.bitwise_not(shiftedDigitBox)

            # Prepare the box for the model to predict the digit
            preparedImage = shiftedDigitBox.reshape(-1, 28, 28, 1)
            preparedImage = preparedImage.astype('float32')
            preparedImage = preparedImage / 255

            # Get the prediction
            prediction = model.predict(preparedImage)

            # Overwrite with the index of the class with the maximum probability
            digits[i][j] = np.argmax(prediction[0]) + 1

    return digits


def overlaySolution(image, fullSolution, digits):

    # Find the height and width of each box
    boxHeight = image.shape[0] // 9
    boxWidth = image.shape[1] // 9

    # Loop through each box
    for i in range(9):
        for j in range(9):

            # This box already had a digit continue to the next box
            if digits[i][j] != 0:
                continue

            # Calculate parameters used to write the solution to the Sudoku Puzzle
            font = cv2.FONT_HERSHEY_SIMPLEX
            digitString = str(fullSolution[i][j])
            xOffset = boxWidth // 15
            yOffset = boxHeight // 15
            (textHeight, textWidth), baseline = cv2.getTextSize(
                digitString, font, fontScale=1, thickness=3)
            fontScale = 0.6 * min(boxHeight, boxWidth) / \
                max(textHeight, textWidth)
            textHeight *= fontScale
            textWidth *= fontScale
            xBottomLeftCorner = boxWidth*j + \
                math.floor((boxWidth - textWidth) / 2) + xOffset
            yBottomLeftCorner = boxHeight * \
                (i+1) - math.floor((boxHeight - textHeight) / 2) + yOffset

            # Overlay the solution on the Sudoku Puzzle
            overlayedSolution = cv2.putText(image, digitString, (xBottomLeftCorner, yBottomLeftCorner), font,
                                            fontScale, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    return overlayedSolution


def perspectiveTransform(corners):

    TL = corners[0]
    TR = corners[1]
    BL = corners[2]
    BR = corners[3]

    # TL - TR
    # |     |
    # BL - BR

    # Lengths of the 4 sides
    lengthTLTR = math.sqrt(((TL[0] - TR[0]) ** 2) + ((TL[1] - TR[1]) ** 2))
    lengthTRBR = math.sqrt(((TR[0] - BR[0]) ** 2) + ((TR[1] - BR[1]) ** 2))
    lengthBRBL = math.sqrt(((BR[0] - BL[0]) ** 2) + ((BR[1] - BL[1]) ** 2))
    lengthBLTL = math.sqrt(((BL[0] - TL[0]) ** 2) + ((BL[1] - TL[1]) ** 2))

    # Height and width of the Sudoku Puzzle
    height = max(int(lengthBLTL), int(lengthTRBR))
    width = max(int(lengthTLTR), int(lengthBRBL))

    # Construct Sudoku Puzzle board destination array
    destinationArray = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype="float32")

    # Perform the perspective transform
    value = cv2.getPerspectiveTransform(corners, destinationArray)

    return value


def solve(frame, model):

    # Declare iterations as a global variable
    global iterations

    # Make a copy of the Sudoku Puzzle to be used later
    originalCopy = np.copy(frame)

    # Preform some pre-processing to the image
    image = imagePreProcessing(frame)

    # Find the best contour in the image (the Sudoku Puzzle)
    bestContour = findBestContour(image)

    # Return the original image if there is no Sudoku Puzzle yet
    if bestContour is None:
        iterations = 0
        return originalCopy

    # Find the 4 corners of the best contour
    fourCorners = fourCornersOfBestContour(bestContour)

    # Return the original image if there is no Sudoku Puzzle yet
    if fourCorners is None:
        iterations = 0
        return originalCopy

    # Locate the top left, top right, bottom left, & bottom right corners
    corners = locateCorners(fourCorners)

    # Copy the corners to be used later
    cornersCopy = copy.deepcopy(corners)

    # Return the original image if the 4 corners of the best contour are not square
    if not square(corners):
        iterations = 0
        return originalCopy

    # Grab the Sudoku Puzzle board
    sudokuPuzzleBoard = grabSudokuPuzzleBoard(corners, originalCopy)

    # Preform some processing to the Sudoku Puzzle board
    processedSudokuPuzzleBoard = processSudokuPuzzleBoard(sudokuPuzzleBoard)

    # Extract & classify the digits of the Sudoku Puzzle
    digits = extractAndClassifyDigits(processedSudokuPuzzleBoard, model)

    # Initialize a list for the full Sudoku Puzzle solution
    fullSolution = copy.deepcopy(digits)

    # If it has been at least 3 iterations
    if iterations > 2:
        # Solve the Sudoku Puzzle
        try:
            sudokuSolver.solveSudokuPuzzle(fullSolution)
        except:
            pass

        # Return the original image if there is no valid Sudoku Puzzle solution yet
        if fullSolution == digits:
            return originalCopy

    # Increment the iterations counter & return the original image if has not yet been 3 iterations
    else:
        iterations += 1
        return originalCopy

    # Overlay the solution on the Sudoku Puzzle
    solvedSudokuPuzzle = overlaySolution(
        sudokuPuzzleBoard, fullSolution, digits)

    # Apply an Inverse Perspective Transform and overlay the solution on the original image
    sudokuSolution = cv2.warpPerspective(
        solvedSudokuPuzzle, perspectiveTransform(cornersCopy), (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    finalSudokuSolution = np.where(sudokuSolution.sum(
        axis=-1, keepdims=True) != 0, sudokuSolution, originalCopy)

    return finalSudokuSolution
