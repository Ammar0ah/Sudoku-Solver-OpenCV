import operator
import cv2 as cv
import joblib
import numpy as np
import os
import feature_extraction
import sudukoSolver
import pySudoku

from sudoku import Sudoku
import digit_detection
from skimage.feature import hog


def basic_preprocess(img):
    gray = img
    if len(img.shape) > 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 2)
    threshold = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dilated = cv.dilate(threshold, kernel)
    erod = cv.erode(dilated, kernel, iterations=2)
    return erod


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


def find_contours(img):
    contours, hierarchy = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours


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


def find_grid_cells(img, isVideo):
    big_stride = img.shape[:1]
    stride = int(big_stride[0] / 9)
    squares = []

    for i in range(9):
        for j in range(9):
            p1 = (i * stride + 5, j * stride + 5)
            p2 = ((i + 1) * stride, (j + 1) * stride)
            square = img[p1[0]:p2[0], p1[1]:p2[1]]
            if not isVideo:
                square = cv.rectangle(square, (0, 0), square.shape, (0, 0, 0), 5)
            squares.append(square)
    return squares


def extract_digits(img, isVideo):
    squares = find_grid_cells(img, isVideo)
    digits = []
    valid_descriptors = []
    for i, digit in enumerate(squares):
        w, h = digit.shape
        if w > 20 and h > 20:
            # cv.imshow('diog', digit)
            # digit = cv.pyrUp(digit)
            digit = cv.resize(digit, (28, 28), interpolation=cv.INTER_AREA)

            descriptors = hog(digit, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(7, 7))
            non_empty = np.count_nonzero(descriptors)
            if non_empty < 20:
                digits.append(0)
            else:
                digits.append(descriptors)
            cv.imwrite(f'imgs/digits/digit{i + 1}.jpg', digit)
    return digits, valid_descriptors, squares


def create_board(board: list, squares):
    result = []
    images = load_images_from_folder()
    # clf = joblib.load("knn_model.pkl")
    final_board = []
    for i, digit in enumerate(board):
        if isinstance(digit, np.ndarray):
            # digit1 = digit.reshape(1, -1)
            # nbr = clf.predict(digit1)
            # predict_proba = clf.predict_proba(digit1)
            # acc = predict_proba[0][nbr]
            res = []
            # print('accuracy',acc,nbr[0])
            # if acc[0] < 0.8:
            for image in images:
                match = digit_detection.flann_matcher(image, squares[i], True)
                res.append(match)
            highest = np.array(res).argmax()
            print(res)
            print('highest', highest + 1)
            result.append(highest + 1)
            # else:
            #     result.append(str(nbr[0]))
        else:
            result.append(0)
    # result = np.array(result).reshape((9, 9))
    return result


def create_board_knn_cv(board: list, squares):
    result = []
    images = load_images_from_folder()
    knn = cv.ml.KNearest_load('KNN_Trained_Model.xml')

    # clf = joblib.load("knn_model.pkl")
    final_board = []
    for i, digit in enumerate(board):
        if isinstance(digit, np.ndarray):
            res = []
            img = cv.resize(squares[i], (28, 28), interpolation=cv.INTER_CUBIC)
            hog_ = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
            hog_ = np.array(hog_).reshape(1, -1)
            hog_ = np.float32(hog_)

            ret, res_knn, neighbours, dist = knn.findNearest(hog_, k=3)
            result.append(int(res_knn[0]))
        else:
            result.append(0)
    # result = np.array(result).reshape((9, 9))
    return result


def create_board_knn(board, squares):
    result = []
    images = load_images_from_folder()
    clf = joblib.load("knn_model.pkl")
    final_board = []
    for i, digit in enumerate(board):
        if isinstance(digit, np.ndarray):
            digit1 = digit.reshape(1, -1)
            nbr = clf.predict(digit1)
            predict_proba = clf.predict_proba(digit1)
            acc = predict_proba[0][nbr]
            res = []
            print('accuracy', acc, nbr[0])
            if acc[0] < 0.8:
                for image in images:
                    match = digit_detection.flann_matcher(image, squares[i], True)
                    res.append(match)
                highest = np.array(res).argmax()
                print(res)
                print('highest', highest + 1)
                result.append(highest + 1)
            else:
                result.append(str(nbr[0]))
        else:
            result.append(0)
    # result = np.array(result).reshape((9, 9))
    return result


#
def solve_sudoku(board):
    board = np.array(board).reshape((9, 9))
    print(board)
    board = board.tolist()
    puzzle = Sudoku(3, 3, board=board)
    print(puzzle)
    return puzzle.solve()


def load_images_from_folder():
    images = []
    paths = ['imgs/templates1/1.png',
             'imgs/templates1/2.png',
             'imgs/templates1/3.png',
             'imgs/templates1/4.png',
             'imgs/templates1/5.png',
             'imgs/templates1/6.png',
             'imgs/templates1/7.png',
             'imgs/templates1/8.png',
             'imgs/templates1/9.png']

    paths1 = ['imgs/templates1/1.jpg',
              'imgs/templates1/2.jpg',
              'imgs/templates1/3.jpg',
              'imgs/templates1/4.jpg',
              'imgs/templates1/5.jpg',
              'imgs/templates1/6.jpg',
              'imgs/templates1/7.jpg',
              'imgs/templates1/8.jpg',
              'imgs/templates1/9.jpg']
    for path in paths1:
        img = cv.imread(path)
        if img is not None:
            images.append(img)
    return images


def solve_paint_sudoku(sudoku_unsolved, shape):
    sudoku_image = np.zeros(shape, np.uint8)
    y = -1
    x = 0
    sudoku_unsolved = np.array(sudoku_unsolved)
    sudoku_unsolved = sudoku_unsolved.astype(int)
    sudoku_unsolved = np.asarray(sudoku_unsolved)
    posArray = np.where(sudoku_unsolved > 0, 0, 1)
    board = np.array_split(sudoku_unsolved, 9)
    print(board)
    try:
        sudukoSolver.solve(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList * posArray
    print(solvedNumbers)
    factor = shape[0] // 9
    for num in sudoku_unsolved:
        if (x % 9) == 0:
            x = 0
            y += 1
        textX = int(factor * x + factor / 2)
        textY = int(factor * y + factor / 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        if num != '0':
            cv.putText(sudoku_image, str(num), (textX, textY), font, 1, (255, 255, 255), 6)
        x += 1

    for i in range(10):
        cv.line(sudoku_image, (0, factor * i), (shape[1], factor * i), (255), 2, 2)
        cv.line(sudoku_image, (factor * i, 0), (factor * i, shape[0]), (255), 2, 2)

    return board, sudoku_image
