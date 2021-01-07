import math
import operator

import cv2 as cv
import numpy as np
import utils
import sudoku_solver
from threading import Thread
import digit_detection
from multiprocessing import Pool

cap = cv.VideoCapture(0)
count = 0

if __name__ == '__main__':
    to_soclve = []

    while True:
        ret, frame = cap.read()
        orig_img = frame.copy()
        # img = cv.pyrUp(img)
        dilated = utils.basic_preprocess(frame)
        # Contours
        contours = utils.find_contours(dilated)
        if len(contours) > 0:
            img, crop_rect = utils.draw_corners(dilated, contours)
            img = utils.find_grid(img, crop_rect)
            digits, numbers_descriptors, squares = utils.extract_digits(img, True)
            cv.imshow('img before', img)

            # if count % 15 == 0:

            templ = cv.imread('polished.jpg', 0)
            images = utils.load_images_from_folder()
            result = []
            match = digit_detection.flann_matcher(img, templ, False)
            if match > 30:
                print(count)

            if cv.waitKey(40) == 27:

                board = utils.create_board_knn(digits, squares)
                to_solve = np.zeros(img.shape)
                board, to_solve, result = sudoku_solver.solve_sudoku(board, to_solve)
                print(to_solve)
                cv.imshow('img after', to_solve)

                if result:
                    break
            count += 1
    cap.release()
    cv.imshow('solved', to_solve)
    print(result)
    cv.waitKey(0)
    # cv.destroyAllWindows()
