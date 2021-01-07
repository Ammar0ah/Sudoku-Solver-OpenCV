import joblib

import utils

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

from skimage.feature import hog
from sklearn.svm import LinearSVC
import cv2 as cv
import numpy as np

import pandas as pd
import os
import feature_extraction
from scipy import spatial


def prepare_imgs(imgs, csv=True):
    res = []
    for i in range(len(imgs)):
        if csv:
            current_img = imgs.iloc[i].to_numpy().reshape((28, 28))
        else:
            current_img = np.array(imgs[i], dtype='float32')

        current_img = np.array(current_img, dtype=np.uint8)

        blur = cv.GaussianBlur(current_img, (5, 5), 2)
        threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 2)
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        dilated = cv.dilate(threshold, kernel)
        erod = cv.erode(dilated, kernel, iterations=1)
        roi_hog_fd = hog(erod, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        res.append(roi_hog_fd)
    # if csv:
    #     res = np.array(res, dtype='float32')
    #     res = res.reshape(res.shape[0], res.shape[1] * res.shape[2])
    return res


def classify_knn():
    training_data = pd.read_csv("train.csv")
    training_imgs, testing_imgs, training_labels, testing_labels = train_test_split(training_data.drop("label", axis=1),
                                                                                    training_data["label"],
                                                                                    train_size=0.8,
                                                                                    test_size=0.2)

    # Smoothing images
    final_training_imgs = prepare_imgs(training_imgs)
    final_testing_imgs = prepare_imgs(testing_imgs)
    # Training SVC
    features = np.array(final_training_imgs, 'float64')
    print(np.array(training_labels).shape, features.shape)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(features, training_labels.values.ravel())
    print("The classifier's score is {}".format(classifier.score(final_testing_imgs, testing_labels)))
    joblib.dump(classifier, "knn_model1.pkl", compress=3)


def predict(img, clf):
    img = cv.imread(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 2)
    threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv.dilate(threshold, kernel)
    erod = cv.erode(dilated, kernel, iterations=1)
    roi = cv.resize(erod, (28, 28), interpolation=cv.INTER_AREA)

    #  roi = cv.dilate(roi, (1, 1))
    # Calculate the HOG featureshog(training_digit, orientations=8, pixels_per_cell=(4,4), cells_per_block=(7, 7))
    roi_hog_fd = hog(roi, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(7, 7))
    roi_hog_fd = roi_hog_fd.reshape(1, -1)
    nbr = clf.predict(roi_hog_fd)

    return str(nbr[0])


def classify_linearSVC():
    dataset = datasets.load_digits()
    features = np.array(dataset.data, dtype='int16')
    labels = np.array(dataset.target, dtype='int')
    print(features.shape, labels.shape)
    list_hog_fd = []
    for feature in features:
        print(feature.shape, len(feature))
        pad = np.zeros((784,))
        pad[:64] = feature
        fd = hog(pad.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')
    clf = LinearSVC()
    clf.fit(hog_features, labels)
    joblib.dump(clf, "linear_svc.pkl", compress=3)


def knn_matcher():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(1, 21):
        for j in range(1, 10):
            folder = f'dataset/{i}/{j}.png'
            img = cv.imread(folder)
            # cv.imshow(f'imgg', img)
            # cv.waitKey(0)
            img = cv.resize(img, (28, 28), interpolation=cv.INTER_CUBIC)
            descriptor = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
            X_train.append(descriptor)
            # print(j)
            y_train.append(j)

    for i in range(1, 10):
        img = cv.imread(f'dataset/22/{i}.png')
        img = cv.bitwise_not(img)
        # cv.imshow(f'imgg', img)
        # cv.waitKey(0)
        img = cv.resize(img, (28, 28), interpolation=cv.INTER_CUBIC)
        descriptor = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        X_test.append(descriptor)
        y_test.append(j)

        img = cv.imread(f'dataset/23/{i}.png')
        img = cv.bitwise_not(img)
        # cv.imshow(f'imgg', img)
        # cv.waitKey(0)
        img = cv.resize(img, (28, 28), interpolation=cv.INTER_CUBIC)
        descriptor = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        X_test.append(descriptor)
        y_test.append(j)
        img = cv.imread(f'dataset/21/{i}.png')
        img = cv.bitwise_not(img)
        # cv.imshow(f'imgg', img)
        # cv.waitKey(0)
        img = cv.resize(img, (28, 28), interpolation=cv.INTER_CUBIC)
        descriptor = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        X_test.append(descriptor)
        y_test.append(j)

    y_train = np.array(y_train, 'float32')
    X_train = np.array(X_train, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    X_test = np.array(X_test, dtype='float32')

    print(y_train.shape, X_train.shape,X_test.shape,y_test.shape)
    knn = cv.ml.KNearest_create()
    knn.train(X_train, cv.ml.ROW_SAMPLE, y_train)
    ret, result, neighbours, dist = knn.findNearest(X_test, k=3)
    # Now we check the accuracy of classification
    # For that, compare the result with test_y_test and check which are wrong
    matches = result == y_test
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print(accuracy)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    print("The classifier's score is {}".format(classifier.score(X_test,y_test)))
    joblib.dump(classifier, "knn_model1.pkl", compress=3)


    knn.save('KNN_Trained_Model.xml')
    return knn

def flann_matcher(img, templ, digits):
    if digits:
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_CUBIC)
        templ = cv.resize(templ, (64, 64), interpolation=cv.INTER_CUBIC)
    else:
        templ = cv.resize(templ, img.shape, interpolation=cv.INTER_CUBIC)

    print(templ.shape)
    fast = cv.FastFeatureDetector_create()
    kp1 = fast.detect(img)
    kp2 = fast.detect(templ)
    good = []
    try:
        # Match descriptors.
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.compute(img, kp1)
        kp2, des2 = sift.compute(templ, kp2)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                # matchesMask[i] = [1, 0]
                good.append([m])
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv.DrawMatchesFlags_DEFAULT)
        img3 = cv.drawMatchesKnn(img, kp1, templ, kp2, matches, None, **draw_params)

        # print(matchesMask)
        cv.imshow('template', img3)
        # cv.waitKey()
    except Exception as e:
        good.append(0)
        print("Error in matching", e)
    print('accuracy: ', len(good))

    return len(good)


if __name__ == '__main__':
    img = cv.imread('imgs/6.png', 0)
    # templ = cv.imread('polished.jpg', 0)
    # images = utils.load_images_from_folder()
    # result = []
    # match = flann_matcher(img, templ, True)
    # for i, image in enumerate(images):
    #     image = cv.resize(image,(28,28))
    #     result.append(match)
    # highest = np.array(result).argmax()
    # print(highest)
    # print(match)
    # knn = knn_matcher()
    img = cv.resize(img, (28, 28), interpolation=cv.INTER_CUBIC)

    hog = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    hog = np.array(hog).reshape(1,-1)

    hog = np.float32(hog)
    print(np.array(hog).shape)
    knn = cv.ml.KNearest_load('KNN_Trained_Model.xml')
    knn_sci = joblib.load("knn_model1.pkl")
    prd = knn_sci.predict(hog)
    print('pred',prd)
    ret, result, neighbours, dist= knn.findNearest(hog,k=3)

    print(result)
