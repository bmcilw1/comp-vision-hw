# Brian McIlwain
# Computer Vision
# Course Project - Obstacle detection
# This program takes in an image, filters out floor data, and finally highlights potential obstacles

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Remove some noise for Canny edge detection
def generalFilter(img_gray, sigma):
    Gsigma = GaussianFilter(sigma)
    gauss = cv2.filter2D(img_gray, -1, Gsigma)
    return gauss

def gradient(img):
    Sx = np.matrix([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

    Sy = np.matrix([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
    
    # Ix = img * Sx
    Ix = cv2.filter2D(img, -1, Sx)
    Iy = cv2.filter2D(img, -1, Sy)

    orient = np.arctan2(Iy,Ix)

    Ixn = np.zeros(Ix.shape)
    Iyn = np.zeros(Iy.shape)

    # Normalize for presentation
    Ixn = cv2.normalize(src=Ix, dst=Ixn, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    Iyn = cv2.normalize(src=Iy, dst=Iyn, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    cv2.imshow('Ix_Normalized', Ixn)
    cv2.imwrite('course_proj/Ix_Normalized.png', Ixn)
    cv2.imshow('Iy_Normalized', Iyn)
    cv2.imwrite('course_proj/Iy_Normalized.png', Iyn)

    return orient

def calcNormal(img):
    normal = img.copy()
    
    return normal

# Remove floor
def filterFloor(gauss):
    floorless = gauss.copy()

    # Attempt 1 - if the gradient is pointing vertically, remove it
    orient = gradient(gauss)
    
    floorless[((3*np.pi/8 < orient) & (orient < 5*np.pi/8)) | ((-5*np.pi/8 < orient) & (orient < -3*np.pi/8))] = 0

    # Lessen/Remove the left-over stripes
    kernel = np.matrix([[.5],
                        [.5]])
    floorless = cv2.erode(floorless,kernel,iterations = 1)

    # Attempt 2 - Calculate normal, if vertical, remove
    # Problem 1 - how to calculate normal?
    # Problem 2 - how to define the ground plane direction of normal
    #              - Best to do with training data
    # Problem 3 - how to generate sufficient training data?
    #              - Simply not feasable within the scope and 
    #                available resources of this project. 
    #                Too much training data to collect would be required

    return floorless

# Visual for showing detected obstacles in image
def markObstaclesRed(img, gauss, edges, thresh):
    img = img.copy()

    # Shade regions close enough
    img[gauss > thresh] = [0,0,255]

    # Fill in shaded regions according to edge image
    y, x, z = img.shape
    for i in range(1, x-1):
        for j in range(1, y-1):
            if img[j,i,2] == 255:
                if edges[j+1, i  ] < 1 and gauss[j+1, i  ] != 0:
                    img.itemset((j+1, i  , 0), 0)
                    img.itemset((j+1, i  , 1), 0)
                    img.itemset((j+1, i  , 2), 255)
                if edges[j  , i+1] < 1 and gauss[j  , i+1] != 0:
                    img.itemset((j  , i+1, 0), 0)
                    img.itemset((j  , i+1, 1), 0)
                    img.itemset((j  , i+1, 2), 255)
                if edges[j+1, i+1] < 1 and gauss[j+1, i+1] != 0:
                    img.itemset((j+1, i+1, 0), 0)
                    img.itemset((j+1, i+1, 0), 0)
                    img.itemset((j+1, i+1, 0), 255)

    # Fill in shaded regions according to edge image - going backwards
    for i in range(x-1, 1, -1):
        for j in range(y-1, 1, -1):
            if img[j,i,2] == 255:
                if edges[j-1, i  ] < 1 and gauss[j-1, i  ] != 0:
                    img.itemset((j-1, i  , 0), 0)
                    img.itemset((j-1, i  , 1), 0)
                    img.itemset((j-1, i  , 2), 255)
                if edges[j  , i-1] < 1 and gauss[j  , i-1] != 0:
                    img.itemset((j  , i-1, 0), 0)
                    img.itemset((j  , i-1, 1), 0)
                    img.itemset((j  , i-1, 2), 255)
                if edges[j-1, i-1] < 1 and gauss[j-1, i-1] != 0:
                    img.itemset((j-1, i-1, 0), 0)
                    img.itemset((j-1, i-1, 0), 0)
                    img.itemset((j-1, i-1, 0), 255)

    return img

# Return 2D map of free space
def get2DMap(filtered, thresh):
    # Get max of columns
    max_vector = np.amax(filtered, axis=0)

    # Smooth local max according to neighbors
    max_vector = cv2.medianBlur(max_vector, 5)

    # Prep 2D map
    mp = np.zeros((256, filtered.shape[1]))

    # Set max value along axis
    for i in range(0, filtered.shape[1]):
        #if filtered[max_vector[i],i] > thresh:
        mp.itemset((max_vector[i],i), 255)

    return mp

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
gauss = generalFilter(img_gray, 1)
cv2.imshow('Gaussian filtered image', gauss)
cv2.imwrite('course_proj/Gaussianfilteredimg.png', gauss)

# Get edge image
edges = cv2.Canny(gauss, 100, 100)
cv2.imshow('Canny edge detection', edges)
cv2.imwrite('course_proj/Cannyedgedetection.png', edges)

# Filter out floor
filter_img = filterFloor(gauss)
cv2.imshow('Removed Floor', filter_img)
cv2.imwrite('course_proj/FilteredFloor.png', filter_img)

# Create 2D map of obstacle-free space
mp = get2DMap(filter_img, thresh)
cv2.imshow('2D Map', mp)
cv2.imwrite('course_proj/Map2D.png', mp)

# Visually Mark obstacles
obst = markObstaclesRed(img, filter_img, edges, thresh)
cv2.imshow('Marked obstacles', obst)
cv2.imwrite('course_proj/MarkedObstacles.png', obst)

cv2.waitKey(0)
cv2.destroyAllWindows()
