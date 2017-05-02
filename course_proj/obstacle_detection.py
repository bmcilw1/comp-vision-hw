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

# Visual for showing detected obstacles in image
def markObstaclesRed(img, gauss, edges, thresh):
    img = img.copy()

    # Shade regions close enough
    img[np.where(gauss > thresh)] = [0,0,255]

    # Fill in shaded regions according to edge image
    y, x, z = img.shape
    for i in range(1, x-1):
        for j in range(1, y-1):
            if img[j,i,2] == 255:
                if edges[j+1, i  ] < 1 and gauss[j+1, i  ] != 0:
                    img [j+1, i  ] = [0,0,255]
                if edges[j  , i+1] < 1 and gauss[j  , i+1] != 0:
                    img [j  , i+1] = [0,0,255]
                if edges[j+1, i+1] < 1 and gauss[j+1, i+1] != 0:
                    img [j+1, i+1] = [0,0,255]

    # Fill in shaded regions according to edge image - going backwards
    for i in range(x-1, 1, -1):
        for j in range(y-1, 1, -1):
            if img[j,i,2] == 255:
                if edges[j-1, i  ] < 1 and gauss[j-1, i  ] != 0:
                    img [j-1, i  ] = [0,0,255]
                if edges[j  , i-1] < 1 and gauss[j  , i-1] != 0:
                    img [j  , i-1] = [0,0,255]
                if edges[j-1, i-1] < 1 and gauss[j-1, i-1] != 0:
                    img [j-1, i-1] = [0,0,255]

    return img

# Shade along vertical max
def markObstaclesGray(img, gauss, edges, thresh):
    img = img.copy()
    y, x, z = img.shape

    # Fill in shaded regions according to edge image
    for i in range(1, x-1):
        for j in range(1, y-1):
            if gauss[j,i] > thresh:
                if edges [j,i] == 1:
                    break
                if edges [j+1,i] < 1 and gauss[j+1,i] != 0:
                    gauss[j,i] = gauss[j+1,i] = max(gauss[j+1,i], gauss[j,i])
                if edges [j,i+1] < 1 and gauss[j,i+1] != 0:
                    gauss[j,i] = gauss[j+1,i] = max(gauss[j,i+1], gauss[j,i])
                if edges [j+1,i+1] < 1 and gauss[j+1,i+1] != 0:
                    gauss[j,i] = gauss[j+1,i] = max(gauss[j+1,i+1], gauss[j,i])

    # Fill in shaded regions according to edge image - going backwards
    for i in range(x-1, 1, -1):
        for j in range(y-1, 1, -1):
            if gauss[j,i] > thresh:
                if edges [j,i] == 1:
                    break
                if edges [j-1,i] < 1 and gauss[j-1,i] != 0:
                    gauss[j,i] = gauss[j-1,i] = max(gauss[j-1,i], gauss[j,i])
                if edges [j,i-1] < 1 and gauss[j  , i-1] != 0:
                    gauss[j,i] = gauss[j-1,i] = max(gauss[j,i-1], gauss[j,i])
                if edges [j-1,i-1] < 1 and gauss[j-1,i-1] != 0:
                    gauss[j,i] = gauss[j-1,i] = max(gauss[j-1,i-1], gauss[j,i])

    return gauss

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
obst = markObstaclesRed(img, gauss, edges, thresh)
cv2.imshow('Marked obstacles', obst)
cv2.imwrite('course_proj/MarkedObstacles.png', edges)

# Create 2D map of obstacle-free space

cv2.waitKey(0)
cv2.destroyAllWindows()
