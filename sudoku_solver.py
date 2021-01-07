import cv2
import numpy as np
import utils
import pySudoku
"""
This module finds the solution of a given sudoku problem
Code credits: Tim Ruscica
More info: https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/
Example input board
board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,2],
    [1,2,0,0,0,7,4,0,0],
    [0,4,9,2,0,6,0,0,7]
]
"""

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            bo[row][col] = 0
    return False

def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True

def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col
    return None



def solve_sudoku(sudoku_unsolved, sudoku_image):
    shape = sudoku_image.shape
    y = -1
    x = 0
    sudoku_unsolved = np.array(sudoku_unsolved)
    sudoku_unsolved = sudoku_unsolved.astype(int)
    sudoku_unsolved = np.asarray(sudoku_unsolved)
    posArray = np.where(sudoku_unsolved > 0, 0, 1)
    board = np.array_split(sudoku_unsolved, 9)
    print(board)
    res = False
    try:
       # res= solve(board)
       res = pySudoku.solve(board)
       # print('result',puzzle,type(puzzle))

    except:
        pass
    print('board for nsereat',np.array(board).reshape((9,9)))
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList * posArray
    print(solvedNumbers)
    factor = shape[0] // 9
    for num in sudoku_unsolved:
        if (x % 9) == 0:
            x = 0
            y += 1
        textX = int(factor * x + factor / 2)
        textY = int(factor * y + factor / 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if num != '0':
            cv2.putText(sudoku_image, str(num), (textX, textY), font,1, (255), 6)
        x += 1

    for i in range(10):
        cv2.line(sudoku_image, (0, factor * i), (shape[1], factor * i), (255), 2, 2)
        cv2.line(sudoku_image, (factor * i, 0), (factor * i, shape[0]), (255), 2, 2)

    return board, sudoku_image,res


