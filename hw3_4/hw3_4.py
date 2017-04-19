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

    return 0

def hw3(i1, i2):
    c1, c1Show = harris(i1, 5, 5, .04, .005)
    c2, c2Show = harris(i2, 5, 5, .04, .005)

    # Show harris corner images
    cv2.imshow("Corner img1", c1Show)
    cv2.imshow("Corner img2", c2Show)

    # Get array of coordinates for corners
    cord1 = getCornerCoordinates(c1, .02)
    cord2 = getCornerCoordinates(c2, .02)

    # determine most similar corners
    match = matchCorners(cord1, cord2, i1, i2)

    return

# Provided by instructor
def construct_openCVKeyPtList(corners, keyptsize=1):
    keyPtsList = []
    for i in range(len(corners)):
        #corners[i][1] and corners[i][0] store the column and row index of the corner
        keyPtsList.append(cv2.KeyPoint(x=corners[i][1], y=corners[i][0], _size=keyptsize))
    return keyPtsList

# Provided by instructor
def construct_openCVDMatch(corners1, corners2, cornerMatches):
    dmatch = list()
    for i in range(len(cornerMatches)):
        #cornerMatches[i][0] and cornerMatches[i][1] store the indices of corresponded corners, respectively
        #cornerMatches[i][2] stores the corresponding ZNCC matching score
        c_match = cv2.DMatch(cornerMatches[i][0], cornerMatches[i][1], cornerMatches[i][2])
        dmatch.append(c_match)    
    return dmatch
    
# Provided by instructor
def draw_matches(img1, img2, corners1, corners2, matches):
    keyPts1 = construct_openCVKeyPtList(corners1)
    keyPts2 = construct_openCVKeyPtList(corners2)
    dmatch = construct_openCVDMatch(corners1, corners2, matches)
    matchingImg = cv2.drawMatches(img1, keyPts1, img2, keyPts2, dmatch, None)
    cv2.imwrite('cornerMatching.png', matchingImg)


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
