# Brian McIlwain
# Computer Vision
# Homework 1

import cv2
from matplotlib import pyplot as plt

# Problem 1: Write a function, called myHEQ_RGB(img), to apply histogram equalization directly on a color image.
def myHEQ_RGB(img):
    # Part 1: Split the RGB channels into three, each of which is a grayscale image;
    bImg,gImg,rImg = cv2.split(img)

    # Part 2: Apply equalization on each individual channel;
    bEqu = cv2.equalizeHist(bImg)
    gEqu = cv2.equalizeHist(gImg)
    rEqu = cv2.equalizeHist(rImg)

    # Part 3: Merge the enhanced new channels;
    newImg = cv2.merge([bEqu, gEqu, rEqu])

    # Part 4: Show the result in a window named “HEQ_RGB”
    cv2.imshow('HEQ_RGB', newImg)
    cv2.imwrite('HEQ_RGB.png', newImg)
    return

# Problem 2: Write a function, called myHEQ_YCRCB(img), to apply histogram equalization on the Y channel of a color image.
def myHEQ_YCRCB(img):
    # Part 1: Convert the RGB representation to YCrCb representation;
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)    cv2.imshow('YCRCB', img_YCrCb)

    # Part 2: Apply equalization on the Y channel;
    yImg,crImg,cbImg = cv2.split(img)
    yEqu = cv2.equalizeHist(yImg)

    # Part 3: Merge the new Y channel with the other two channels;
    newImg = cv2.merge([yEqu, crImg, cbImg])

    # Part 4: Convert the image back to RGB mode.
    img_new = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)

    # Part 5: Show the result in a window named “HEQ_YCRCB”
    cv2.imshow('HEQ_YCRCB', newImg)
    cv2.imwrite('HEQ_YCRCB.png', newImg)
    return

img = cv2.imread('HW1Description/TestImages/Castle_badexposure.jpg')
cv2.imshow('DEFAULT', img)

myHEQ_RGB(img)
myHEQ_YCRCB(img)

cv2.waitKey(0)
cv2.destroyAllWindows()