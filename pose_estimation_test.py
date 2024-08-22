import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import time
import sys
import glob


def findChessboardStatic(img):
    chessboard_size = (12, 8)
    image = cv.imread(img)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cv.imshow(image)
    cv.waitKey(0)

    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    print(ret)
    if ret:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv.drawChessboardCorners(image, chessboard_size, corners, ret)
        cv.imshow(image)
        cv.waitKey(0)

    cv.destroyAllWindows()


def findChessboard():
    chessboard_size = (12, 8)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, chessboard_size, corners, ret)

        cv.imshow(frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


# TODO Modify code
""""
    camera calibration used to correct distortion, radial or tangential. Must find 5 distortion coefficients (k1 k2 p1 p2 k3)

    Camera matrix stores parameters of camera: focal length and optical centers
"""


def objectTracking():
    video_capture = cv.VideoCapture(0) 

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    bbox = cv.selectROI("Tracking", frame, False)

    tracker = cv.TrackerCSRT_create()

    tracker.init(frame, bbox)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        ret, bbox = tracker.update(frame)

        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else:
            cv.putText(frame, "Tracking failure detected", (100, 80),
                        cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv.imshow('tracking', frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    video_capture.release()
    cv.destroyAllWindows()



def cameraCalibration():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()


# TODO Modify code
def checkerboard():
    null


def camera():
    source = cv.VideoCapture(0)

    cv.namedWindow('camera feed', cv.WINDOW_NORMAL)

    while cv.waitKey(5) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv.imshow('camera feed', frame)

    source.release()
    cv.destroyWindow('camera feed')


def printImgArr(img):
    np.set_printoptions(threshold=np.inf)

    img = cv.imread(img, 0)

    f = open("output.txt", "w")
    f.write(np.array2string(img))
    f.close()


####    MAIN    ####
# printImgArr('image.png')
#findChessboard()
objectTracking()
