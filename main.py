import math
import operator

import cv2 as cv
import numpy as np
import utils

img = cv.imread('imgs/sudoku.jpg')
orig_img = img.copy()
# img = cv.resize(img, (700, 700))
# img = cv.pyrUp(img)
dilated = utils.basic_preprocess(img)
# Contours
contours, hierarchy = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)
if len(contours) > 0:
    dilated, crop_rect = utils.draw_corners(dilated, contours)

    # cv.imshow('img', img)
    digits = utils.extract_digits(img)
# img = cv.imread('imgs/6.png')
# FAST

fast = cv.FastFeatureDetector_create()
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp,None, color=(255,0,0))

fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

orb = cv.ORB_create()
_,descriptors = orb.compute(img,kp)
print(descriptors)
img3 = cv.drawKeypoints(img, kp,None, color=(255,0,0))
cv.imshow('orb fast',img3)
cv.waitKey(0)
# HOG
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradients)

# cv.imshow('img3',img3)
# cv.imshow('img2',img2)
descriptors = hog.compute(img)
# print(descriptors)

# Features track
corners=  cv.goodFeaturesToTrack(img,100,0.01,10)
for cor in corners:
    x,y = cor[0]
    x = int(x)
    y = int(y)
    cv.rectangle(img,(x-10,y-10),(x+10,y+10),(0,255,0),2)
cv.imshow('featrure track corners',img)



cv.waitKey(0)
