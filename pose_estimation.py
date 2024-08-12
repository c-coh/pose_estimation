import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import time
import sys


def checkerboard():

    # start camera stream ()


def camera():
    source = cv.VideoCapture(0)

    cv.namedWindow('camera feed', cv.WINDOW_NORMAL)

    while cv.waitKey(5) != 27:  # Escape key
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv.imshow('camera feed', frame)

    source.release()
    cv.destroyWindow('camera feed')


# get numeric array corresponding to image file
def printImgArr(img):
    # set print option to show entire array
    np.set_printoptions(threshold=np.inf)

    img = cv.imread(img, 0)

    # write to file
    f = open("output.txt", "w")
    f.write(np.array2string(img))
    f.close()


####    MAIN    ####
# printImgArr('image.png')
camera()
