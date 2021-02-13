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
        cv.imwrite(IMG_NAME + "R_L" + str(angle) + ".png", transformed_img_L)
        cv.imwrite(IMG_NAME + "R_R" + str(angle) + ".png", transformed_img_R)


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
    width = width + 2 * max([matrix[1], matrix[2]])
    height = height + 2 * max([matrix[0], matrix[3]])
    left, top, bottom, right = matrix
    coeffs = find_coeffs([(top, left), (width - top, right),  # 00-0x
                          (width - bottom, height - right), (bottom, height - left)],  # 0y-xy
                         [(0, 0), (width, 0), (width, height), (0, height)])
    image = Image.new('RGBA', (width, height), (255, 0, 0, 0))
    image.paste(img, (max([matrix[1], matrix[2]]), max([matrix[0], matrix[3]])))
    a = image.transform((width, height), Image.PERSPECTIVE, coeffs,
                        Image.BICUBIC)
    a.save(IMG_NAME + "_P" + str(matrix) + ".png")


def perspective_Y(path, values):
    for value in values:
        perspective_transform(path, [0, value, -value, 0])
        perspective_transform(path, [0, -value, value, 0])

def perspective_X(path, values):
    for value in values:
        perspective_transform(path, [value, 0, 0, -value])
        perspective_transform(path, [-value, 0, 0, value])

def perspective_XY(path, values):
    for value in values:
        perspective_transform(path, [value, value, -value, -value])
        perspective_transform(path, [-value, -value, value, value])


shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
os.mkdir(OUTPUT_PATH)

for images in os.listdir(INPUT_PATH):
    FOLDER_PATH = (OUTPUT_PATH + "/" + str(images))[:-4]
    IMG_PATH = FOLDER_PATH + "/" + str(images)
    os.mkdir(FOLDER_PATH)
    shutil.copyfile(INPUT_PATH + "/" + str(images), FOLDER_PATH + '/' + str(images))
    rotate_transform(IMG_PATH, [5, 15, 30])

    # left, top, bottom, right
    perspective_Y(IMG_PATH, [25, 45, 60])
    perspective_X(IMG_PATH, [25, 45, 60])
    perspective_XY(IMG_PATH, [25, 45, 60])
