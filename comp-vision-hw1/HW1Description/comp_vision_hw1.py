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
    return newImg

# Problem 2: Write a function, called myHEQ_YCRCB(img), to apply histogram equalization on the Y channel of a color image.
def myHEQ_YCRCB(img):
    # Part 1: Convert the RGB representation to YCrCb representation;
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # Part 2: Apply equalization on the Y channel;
    yImg,crImg,cbImg = cv2.split(img)
    yEqu = cv2.equalizeHist(yImg)

    # Part 3: Merge the new Y channel with the other two channels;
    newImg = cv2.merge([yEqu, crImg, cbImg])

    # Part 4: Convert the image back to RGB mode.
    newImg = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)

    # Part 5: Show the result in a window named “HEQ_YCRCB”
    cv2.imshow('HEQ_YCRCB', newImg)
    return newImg

# Problem 2: Develop a ROI (Region of Interest) Selection Tool and apply Local Equalization:
def myHEQ_ROI(img):
    # Part 1: Implement a mouse-event handling function to allow user to select a rectangular region of Interest (ROI) in the image.
    newImg = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Part 2: Use “myHEQ_YCRCB()” to equalize the ROI region and get an enhanced patch

    # Part 3: Update the ROI in the image using this new enhanced patch

    # Part 5: Show the final result in a window named “HEQ_ROI”
    cv2.imshow('HEQ_ROI', img)
    return newImg

img = cv2.imread('HW1Description/TestImages/Castle_badexposure.jpg')
cv2.imshow('DEFAULT', img)

cv2.imwrite('HW1Description/HEQ_RGB.png', myHEQ_RGB(img))
cv2.imwrite('HW1Description/HEQ_YCRCB.png', myHEQ_YCRCB(img))
cv2.imwrite('HW1Description/HEQ_ROI.png', myHEQ_ROI(img))

cv2.waitKey(0)
cv2.destroyAllWindows()