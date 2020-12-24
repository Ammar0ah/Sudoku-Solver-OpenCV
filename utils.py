import operator
import cv2 as cv
import numpy as np


def basic_preprocess(img):
    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 2)
    threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv.dilate(threshold, kernel)
    return dilated


def draw_corners(img, contours):
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                                     polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                                 polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                                    polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                                  polygon]), key=operator.itemgetter(1))
    cv.circle(img, (polygon[bottom_left][0][0], polygon[bottom_left][0][1]), 3, (0, 0, 255), 3)
    cv.circle(img, (polygon[top_left][0][0], polygon[top_left][0][1]), 3, (0, 0, 255), 3)
    cv.circle(img, (polygon[bottom_right][0][0], polygon[bottom_right][0][1]), 3, (0, 0, 255), 3)
    cv.circle(img, (polygon[top_right][0][0], polygon[top_right][0][1]), 3, (0, 0, 255), 3)
    crop_rect = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    return img, crop_rect


def distance_between(p1, p2):
    a = np.abs(p1[0] - p2[0])
    b = np.abs(p2[0] - p2[0])
    return a + b


def find_grid(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    side = max([distance_between(top_left, top_right), distance_between(top_left, bottom_left),
                distance_between(top_right, bottom_right),
                distance_between(bottom_right, bottom_left)])
    dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype='float32')
    m = cv.getPerspectiveTransform(src, dst)
    img = cv.warpPerspective(img, m, (int(side), int(side)))
    return img


def find_grid_cells(img):
    big_stride = img.shape[:1]
    stride = int(big_stride[0] / 9)
    squares = []

    for j in range(9):
        for i in range(9):
            p1 = (i * stride + 5, j * stride + 5)
            p2 = ((i + 1) * stride, (j + 1) * stride)
            squares.append((p1, p2))
    return squares


def extract_digits(img):
    squares = find_grid_cells(img)
    digits = []
    sift = cv.SIFT_create()

    for i, square in enumerate(squares):
        p1, p2 = square
        digit = img[p1[0]:p2[0], p1[1]:p2[1]]
        # digit = cv.pyrUp(digit)
        digits.append(digit)
        kp = sift.detect(digit)
        # print(kp)
        # digit = cv.drawKeypoints(digit, kp, outImage=None)
        cv.imwrite(f'imgs/digits/digit{i + 1}.jpg', digit)
    return digits
