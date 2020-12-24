import math
import operator

import cv2 as cv
import numpy as np
import utils

img = cv.imread('imgs/6.png')
fast = cv.FastFeatureDetector_create()
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp,None, color=(255,0,0))

fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
img3 = cv.drawKeypoints(img, kp,None, color=(255,0,0))
cv.imshow('img3',img3)
cv.imshow('img2',img2)