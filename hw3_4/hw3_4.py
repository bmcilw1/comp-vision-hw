# Brian McIlwain
# Comp vision hw3 and hw4
 
import cv2
import numpy as np

def init_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def harris(img, blockSize, kSize, k, cThresh):
    # Perform harris
    c = cv2.cornerHarris(img, blockSize, kSize, k)
    cShow = img.copy()

    # viewing Harris corner detecion adapted from: 
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

    # result is dilated for marking the corners
    #cDial = c.copy()
    cDial = cv2.dilate(c, None)

    # Threshold for corner detection
    cShow[cDial>cThresh*cDial.max()]= 1

    return c, cShow

def getCornerCoordinates(c, cThresh):
    cord = []
    unFilterdcord = []
    ctemp = c.copy()
    cmax = c.max()

    # Fist get all possible corner points
    y, x = c.shape
    for x in range(0, x):
        for y in range(0, y):
            # Check if possible corner
            if (c[y,x] > cThresh*cmax):
                unFilterdcord.append((y,x))
                ctemp[y,x] = c[y,x]

    # NMS- Only keep local maximum pixels in each group for each corner
    kernel = np.ones((21,21),np.uint8)
    cNMS = cv2.erode(c,kernel,iterations = 2)

    # Grab max points
    cmax = cNMS.max()
    y, x = cNMS.shape
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

def padding(img, padSize):
    img_pad = np.zeros((img.shape[0] + 2*padSize,img.shape[1] + 2*padSize), img.dtype)
    img_pad[padSize:img.shape[0]+padSize, padSize:img.shape[1]+padSize] = img

    return img_pad

def score_ZNCC(patch1, patch2):
    p1 = patch1.flatten()
    p2 = patch2.flatten()

    p1Norm = p1 - np.mean(p1) 
    p1Norm = p1Norm / np.linalg.norm(p1Norm)

    p2Norm = p2 - np.mean(p2)
    p2Norm = p2Norm / np.linalg.norm(p2Norm)

    return np.dot(p1Norm, p2Norm)

def matchKeyPts(img1, img2, patchSize, corners1, corners2, maxScoreThresh):
    match = []

    # Padd images
    i1Pd = padding(img1, patchSize)
    i2Pd = padding(img2, patchSize)
    
    '''
    cv2.imshow("Pad 1", i1Pd)
    cv2.imshow("Pad 2", i2Pd)
    '''

    for c1 in corners1:
        # get first patch
        p1 = i1Pd[c1[0]: c1[0]+patchSize, c1[1]: c1[1]+patchSize]
        bestScore = 0;
        bestc2 = ();
        for c2 in corners2:
            # get second patch
            p2 = i2Pd[c2[0]: c2[0]+patchSize, c2[1]: c2[1]+patchSize]
            zncc = score_ZNCC(p1, p2)
            if (zncc > bestScore):
                bestc2 = c2
                bestScore = zncc

        if (bestScore > maxScoreThresh):
            match.append((c1, bestc2))

    return match

def hw3(i1, i2):
    c1, c1Show = harris(i1, 5, 5, .04, .005)
    c2, c2Show = harris(i2, 5, 5, .04, .005)

    # Show harris corner images
    cv2.imshow("Corner img1", c1Show)
    cv2.imshow("Corner img2", c2Show)

    # Get array of coordinates for corners
    corners1 = getCornerCoordinates(c1, .02)
    corners2 = getCornerCoordinates(c2, .02)

    # determine most similar corners
    match = matchKeyPts(i1, i2, 15, corners1, corners2, .98)

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

img1 = cv2.imread('hw3_4/TestingImages/goldengate-02.png')
img2 = cv2.imread('hw3_4/TestingImages/goldengate-03.png')

#cv2.imshow("Original img1", img1)
#cv2.imshow("Original img2", img2)

# convert images to grayscale
img1 = init_image(img1)
img2 = init_image(img2)

hw3(img1, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
