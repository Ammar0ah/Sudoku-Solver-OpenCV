import math
import operator

import cv2 as cv
import numpy as np
import utils

cap = cv.VideoCapture(0)
while True:
    ret,img = cap.read()
    # img = cv.imread('imgs/sudoku3.jpg')
    # orig_img = img.copy()
    img = cv.resize(img, (700, 700))
    # img = cv.pyrUp(img)
    dilated = utils.basic_preprocess(img)
    # Contours
    contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    OT_img = img.copy()

    img, crop_rect = utils.draw_corners(dilated, contours)
    img = utils.find_grid(img,crop_rect)

    cv.imshow('img',img)
    cv.imshow('img11',OT_img)
    digits = utils.extract_digits(img)
    if cv.waitKey(40) == 27:
        break
