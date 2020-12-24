import math
import operator

import cv2 as cv
import numpy as np
import utils


def fast_detector(digit):
    fast = cv.FastFeatureDetector_create()
    # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp = fast.detect(digit, None)
    img2 = cv.drawKeypoints(digit, kp, None, color=(255, 0, 0))

    fast.setNonmaxSuppression(0)
    kp = fast.detect(digit, None)

    orb = cv.ORB_create()
    _, descriptors = orb.compute(digit, kp)
    # print(descriptors)
    return img2


def hog_descritor(digit):
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                           L2HysThreshold, gammaCorrection, nlevels, signedGradients)

    descriptors = hog.compute(digit)
    return descriptors


def features_track(digit):
    corners = cv.goodFeaturesToTrack(digit, 100, 0.01, 10)
    for cor in corners:
        x, y = cor[0]
        x = int(x)
        y = int(y)
        cv.rectangle(digit, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 0), 2)
    return digit


def template_matching(digit):
    index_params = dict(algoritm=0, trees=3)
    search_index = dict(checks=100)

    flann = cv.FlannBasedMatcher(index_params, search_index)
