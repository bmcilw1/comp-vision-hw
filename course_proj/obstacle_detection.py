# Brian McIlwain
# Computer Vision
# Course Project - Obstacle detection
# This program takes in an image, filters out floor data, and finally highlights potential obstacles

import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def GaussianFilter(sigma):
    # Round sigma to prevent dimension mis-match
    halfSize = 3 * round(sigma)
    maskSize = 2 * round(halfSize) + 1 
    mat = np.ones((maskSize,maskSize)) / (float)( 2 * np.pi * (sigma**2))
    xyRange = np.arange(-halfSize, halfSize+1)
    xx, yy = np.meshgrid(xyRange, xyRange)    
    x2y2 = (xx**2 + yy**2)    
    exp_part = np.exp(-(x2y2/(2.0*(sigma**2))))
    mat = mat * exp_part

    return mat

def generalFilter(img):
    Gsigma = GaussianFilter(1)
    gauss = cv2.filter2D(img_gray, -1, Gsigma)
    return gauss

def markObstacles(img, gauss, edges, thresh):
    img = img.copy()

    # Shade regions close enough
    img[np.where(gauss > thresh)] = [0,0,255]

    # Fill in shaded regions according to edge image

    return img


# MAIN
##################################################################
img = cv2.imread('course_proj/images/KinectSnapshot-08-36-54.png')

# Threshold closeness to be considered as a potential obstacle
thresh = 150

# Convert to grayscale
img_gray = grayscale(img)

# Show initial image
cv2.imshow('Original image', img)
cv2.imwrite('course_proj/Originalimg.png', img)

# Filter noise
gauss = generalFilter(img_gray)
cv2.imshow('Gaussian filtered image', gauss)
cv2.imwrite('course_proj/Gaussianfilteredimg.png', gauss)

# Get edge image
edges = cv2.Canny(gauss, 100, 100)
cv2.imshow('Canny edge detection', edges)
cv2.imwrite('course_proj/Cannyedgedetection.png', edges)

# Filter out floor

# Mark obstacles
obst = markObstacles(img, gauss, edges, thresh)
cv2.imshow('Marked obstacles', obst)
cv2.imwrite('course_proj/MarkedObstacles.png', edges)

# Create 2D map of obstacle-free space

cv2.waitKey(0)
cv2.destroyAllWindows()
