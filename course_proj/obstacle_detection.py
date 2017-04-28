# Brian McIlwain
# Computer Vision
# Course Project - Obstacle detection
# This program takes in an image, filters out floor data, and finally highlights potential obstacles

import cv2

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# MAIN
##################################################################
img = cv2.imread('course_proj/images/KinectSnapshot-08-36-54.png')

# Convert to grayscale
img = grayscale(img)

# Show initial image
cv2.imshow('original image', img)

# Filter noise

# Filter out floor

# Mark obstacles

# Create 2D map of obstacle-free space

cv2.waitKey(0)
cv2.destroyAllWindows()
