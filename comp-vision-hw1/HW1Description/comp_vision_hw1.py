# Brian McIlwain
# Computer Vision
# Homework 1

import cv2
from matplotlib import pyplot as plt

# Problem 1
def myHEQ_RGB(img):
    bImg,gImg,rImg = cv2.split(img)

    bEqu = cv2.equalizeHist(bImg)
    gEqu = cv2.equalizeHist(gImg)
    rEqu = cv2.equalizeHist(rImg)

    newImg = cv2.merge([bEqu, gEqu, rEqu])

    cv2.imshow('HEQ_RGB', newImg)
    cv2.waitKey(0)
    return

img = cv2.imread('HW1Description/TestImages/Castle_badexposure.jpg')
cv2.imshow('DEFAULT', img)

myHEQ_RGB(img)

cv2.destroyAllWindows()