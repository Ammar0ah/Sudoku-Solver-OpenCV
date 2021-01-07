import math
import operator
import sudukoSolver
import cv2 as cv
import joblib
import numpy as np
import sudoku_solver
import utils

img = cv.imread('imgs/sudoku7.png')
img = cv.resize(img, (700, 700))
orig_img = img.copy()
# img = cv.pyrUp(img)
dilated = utils.basic_preprocess(img)
# Contours
contours = utils.find_contours(dilated)
img, crop_rect = utils.draw_corners(dilated, contours)
img = utils.find_grid(img, crop_rect)
digits, numbers_descriptors, squares = utils.extract_digits(img, False)
cv.imshow('img before', img)
cv.imwrite('polished.jpg', img)
board = utils.create_board_knn(digits, squares)
# print(board)
board_copy = np.array(board, 'int').reshape(9, 9)
print(board_copy)
to_solve = np.zeros(img.shape)
board, to_solve, res = sudoku_solver.solve_sudoku(board, to_solve)

print(board)
cv.imshow('img', to_solve)
cv.waitKey(0)
