# Brian McIlwain
# Comp vision hw3 and hw4
 
import cv2
import numpy as np

def harris(img, blockSize, kSize, k, thresh):
    c = cv2.cornerHarris(img, blockSize, kSize, k)
    cshow = img.copy()

    # viewing Harris corner detecion
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

    # result is dilated for marking the corners
    cDial = cv2.dilate(c, None)

    # Threshold for an optimal value
    cshow[cDial>thresh*cDial.max()]= 255

    # Show harris corner images
    cv2.imshow("Corners img", cshow)

    return c

def hw3(i1, i2):
    c1 = harris(i1, 2, 3, .04, .007)
    c2 = harris(i2, 2, 3, .04, .007)


    return


##################
#  main

# Load images as grayscale
img1 = cv2.imread('hw3_4/TestingImages/goldengate-02.png', 0)
img2 = cv2.imread('hw3_4/TestingImages/goldengate-03.png', 0)

#cv2.imshow("Original img1", img1)
#cv2.imshow("Original img2", img2)

hw3(img1, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
