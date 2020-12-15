import numpy
import imutils
import cv2 as cv
import os
import shutil
from PIL import Image

INPUT_PATH = "Data/Input"
OUTPUT_PATH = "Data/Output"


def rotate_transform(path, angles):
    original_img = cv.imread(path, cv.IMREAD_UNCHANGED)
    IMG_NAME = path[:-4]
    for angle in angles:
        transformed_img_L = imutils.rotate_bound(original_img, angle)
        transformed_img_R = imutils.rotate_bound(original_img, -angle)
        cv.imwrite(IMG_NAME + "_L" + str(angle) + ".png", transformed_img_L)
        cv.imwrite(IMG_NAME + "_R" + str(angle) + ".png", transformed_img_R)


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def perspective_transform(path, matrix):
    IMG_NAME = path[:-4]
    img = Image.open(path)
    width, height = img.size
    left, top, bottom, right = matrix
    coeffs = find_coeffs([(top, left), (width - top, right),  # 00-0x
                          (width - bottom, height - right), (bottom, height - left)],  # 0y-xy
                         [(0, 0), (width, 0), (width, height), (0, height)])

    img.transform((width, height), Image.PERSPECTIVE, coeffs,
                  Image.BICUBIC).save(IMG_NAME + "_P" + str(matrix) + ".png")


shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
os.mkdir(OUTPUT_PATH)

for images in os.listdir(INPUT_PATH):
    FOLDER_PATH = (OUTPUT_PATH + "/" + str(images))[:-4]
    IMG_PATH = FOLDER_PATH + "/" + str(images)
    os.mkdir(FOLDER_PATH)
    shutil.copyfile(INPUT_PATH + "/" + str(images), FOLDER_PATH + '/' + str(images))
    rotate_transform(IMG_PATH, [5, 15, 30])
    perspective_transform(IMG_PATH, [0, 50, 0, 0])
