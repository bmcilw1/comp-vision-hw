# Brian McIlwain
# Comp vision hw3 and hw4
 
import cv2
import numpy as np

def harris(img, blockSize, kSize, k, cThresh):
    # Convert to grayscale and perform harris
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    c = cv2.cornerHarris(gray, blockSize, kSize, k)
    cShow = img.copy()

    # viewing Harris corner detecion adapted from: 
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

    # result is dilated for marking the corners
    #cDial = c.copy()
    cDial = cv2.dilate(c, None)

    # Threshold for corner detection
    cShow[cDial>cThresh*cDial.max()]= [0,0,255]

    return c, cShow

def getCornerCoordinates(c, cThresh):
    cord = []
    unFilterdcord = []
    ctemp = c.copy()
    cmax = c.max()

    # Fist get all possible corner points
    x, y = c.shape
    for x in range(0, x):
        for y in range(0, y):
            # Check if possible corner
            if (c[y,x] > cThresh*cmax):
                unFilterdcord.append((y,x))
                ctemp[y,x] = c[y,x]

    # NMS- Only keep local maximum pixels in each group for each corner
    kernel = np.ones((4,4),np.uint8)
    cNMS = cv2.erode(c,kernel,iterations = 1)

    # Grab max points
    cmax = cNMS.max()
    x, y = cNMS.shape
    for x in range(0, x):
        for y in range(0, y):
            # Check if corner
            if (cNMS[y,x] > cThresh*cmax):
                cord.append((y,x))

    print "len Cord: "
    print len(cord)
    print "len unfiltered Cord: "
    print len(unFilterdcord)

    return np.asarray(cord)

def zncc(cord1, cord2, i1, i2):
    # Compute similarity score for each corner point against every other corner point
    # Return array cord1.size * cord2.size of scores
    matchScore = [] #np.zeros(cord1.size, cord2.size)

    return matchScore

def matchCorners(cord1, cord2, i1, i2):
    # similarity threshold, given in problem
    simThresh = .8

    zncc(cord1, cord2, i1, i2)

    return

def hw3(i1, i2):
    c1, c1Show = harris(i1, 2, 3, .04, .02)
    c2, c2Show = harris(i2, 2, 3, .04, .02)

    # Show harris corner images
    cv2.imshow("Corner img1", c1Show)
    cv2.imshow("Corner img2", c2Show)

    # Get array of coordinates for corners
    cord1 = getCornerCoordinates(c1, .02)
    cord2 = getCornerCoordinates(c2, .02)

    # determine most similar corners
#    match = matchCorners(cord1, cord2, i1, i2)

    return


##################
#  main

# Load images as grayscale
img1 = cv2.imread('hw3_4/TestingImages/goldengate-02.png')
img2 = cv2.imread('hw3_4/TestingImages/goldengate-03.png')

#cv2.imshow("Original img1", img1)
#cv2.imshow("Original img2", img2)

hw3(img1, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
