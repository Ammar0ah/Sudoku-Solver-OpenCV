import math
import operator

import cv2 as cv
import numpy as np
import utils

img = cv.imread('imgs/sudoku3.jpg')
orig_img = img.copy()
img = cv.resize(img, (700, 700))
# img = cv.pyrUp(img)
dilated = utils.basic_preprocess(img)
# Contours
contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)

img, crop_rect = utils.draw_corners(dilated, contours)
img = utils.find_grid(img,crop_rect)
cv.imshow('img',img)
digits = utils.extract_digits(img)

cv.waitKey(0)
