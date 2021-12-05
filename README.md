# SudokuSolverV2
A Computer Vision & Deep Learning Program to Solve Sudoku Puzzles in Real Time.

## How It Works: ##
**STEP 1:** Grab the Sudoku Puzzle board from the webcam image with OpenCV.<br /> 
**STEP 2:** Extract the digits of the Sudoku Puzzle with OpenCV and classify the digits of the puzzle with a Convolutional Neural Network (CNN) in Tensorflow.<br /> 
**STEP 3:** Solve the Sudoku Puzzle recursively with a backtracking algorithm.<br /> 
**STEP 4:** Overlay the solution on the original webcam image with OpenCV.

## Setup: ##
1. Clone this repository
2. Install OpenCV
`pip install opencv-python`
3. Install NumPy
`pip install numpy`
4. Install SciPy
`pip install scipy`
5. Install Tensorflow
`pip install tensorflow`
6. Navigate to the directory with this repository
7. Run main.py
`python main.py`
