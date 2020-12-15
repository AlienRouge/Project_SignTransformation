import numpy as np
import imutils
import cv2 as cv
import os
import shutil

INPUT_PATH = "Data/Input"
OUTPUT_PATH = "Data/Output"


def LR_transform(path, angles):
    original_img = cv.imread(path, cv.IMREAD_UNCHANGED)
    IMG_NAME = path[:-4]
    for angle in angles:
        transformed_img_L = imutils.rotate_bound(original_img, angle)
        transformed_img_R = imutils.rotate_bound(original_img, -angle)
        cv.imwrite(IMG_NAME+"_L"+str(angle)+".png", transformed_img_L)
        cv.imwrite(IMG_NAME+"_R"+str(angle)+".png", transformed_img_R)


shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
os.mkdir(OUTPUT_PATH)

for images in os.listdir(INPUT_PATH):
    FOLDER_PATH = (OUTPUT_PATH + "/" + str(images))[:-4]
    IMG_PATH = FOLDER_PATH + "/" + str(images)
    os.mkdir(FOLDER_PATH)
    shutil.copyfile(INPUT_PATH + "/" + str(images), FOLDER_PATH + '/' + str(images))
    LR_transform(IMG_PATH, [5, 15, 30])

#
