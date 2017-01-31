# Brian McIlwain
# Computer Vision
# Homework 1

import cv2

# Problem 1
def myHEQ_RGB(img):
    cv2.imshow('HEQ_RGB', img)
    cv2.waitKey(0)
    return

img = cv2.imread('HW1Description/TestImages/Castle_badexposure.jpg')
myHEQ_RGB(img)

cv2.destroyAllWindows()