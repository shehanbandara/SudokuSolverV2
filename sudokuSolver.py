# Referencing Tim Ruscica's Implementation
# https://www.techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/

# Function to solve the Sudoku Puzzle using backtracking


def solveSudokuPuzzle(sudokuPuzzle):

    # If the board is fully solved return True
    empty = firstEmptyBox(sudokuPuzzle)
    if not empty:
        return True

    # If the board is not fully solved save the first empty box
    else:
        row, column = empty

    # Loop though possible valid numbers for the empty box
    for i in range(1, 10):

        # If the number is valid at (row, column) add it to the Sudoku Puzzle
        if valid(sudokuPuzzle, i, (row, column)):
            sudokuPuzzle[row][column] = i

            # If the board is fully solved recursively return True
            if solveSudokuPuzzle(sudokuPuzzle):
                return True

            # Reset last number added to 0 if no solution and returned False
            sudokuPuzzle[row][column] = 0

    # If no valid numbers and no solution return False to backtrack
    return False

# Function to return if a value is a valid entry in the Sudoku Puzzle


def valid(sudokuPuzzle, value, position):

    # If value is already in the row return False
    for i in range(len(sudokuPuzzle[0])):
        if sudokuPuzzle[position[0]][i] == value and position[1] != i:
            return False

    # If value is already in the column return False
    for i in range(len(sudokuPuzzle)):
        if sudokuPuzzle[i][position[1]] == value and position[0] != i:
            return False

    # Figure out which of the 9 3x3 sub-boxes you are in
    subBoxX = position[1] // 3
    subBoxY = position[0] // 3

    # If value is already in the 3x3 sub-box return False
    for i in range(subBoxY*3, subBoxY*3 + 3):
        for j in range(subBoxX*3, subBoxX*3 + 3):
            if sudokuPuzzle[i][j] == value and (i, j) != position:
                return False

    return True


# Function to return the first empty box
def firstEmptyBox(sudokuPuzzle):
    for i in range(len(sudokuPuzzle)):
        for j in range(len(sudokuPuzzle[0])):
            if sudokuPuzzle[i][j] == 0:
                return (i, j)
    return None
