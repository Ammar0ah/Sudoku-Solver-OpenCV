import math
import operator

import cv2 as cv
import numpy as np
import utils

img = cv.imread('imgs/sudoku2.jpg')

img = cv.resize(img, (700, 700))

dilated = utils.basic_preprocess(img)
# Contours
contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)

img, crop_rect = utils.draw_corners(dilated, contours)
img = utils.find_grid(img, crop_rect)
# finding numbers squares
cv.imshow('img',img)
digits = utils.extract_digits(img)

cv.waitKey(0)
