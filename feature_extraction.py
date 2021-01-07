import math
import operator

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2 as cv
import numpy as np

import pandas as pd

from skimage.feature import hog


def hog_descriptor(digit):
    orb = cv.ORB_create()
    kp = orb.detect(digit)
    # _, descriptors = orb.compute(digit, kp)
    # img2 = cv.drawKeypoints(digit, kp, None, color=(255, 0, 0))
    roi_hog_fd = hog(digit, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    return None    , roi_hog_fd


def fast_descriptor(digit):
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(digit)
    orb = cv.ORB_create()
    _,descriptors = orb.compute(digit,kp)
    return descriptors

def hog_descriptor_2(digit):
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
