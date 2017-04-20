# Brian McIlwain
# Comp vision hw3 and hw4
 
import cv2
import numpy as np

def init_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def extract_keypts_Harris(img, thresh_Harris=0.005, nms_size=10):
    corners = []
    # Get corner candidates
    c = cv2.cornerHarris(img, 5, 5, .005)

    # Padd boundry
    pad = padding(c, nms_size)

    # Do NMS
    for i in range(nms_size, nms_size+c.shape[0]): # stay within padding range
        for j in range(nms_size, nms_size+c.shape[1]):
            if (pad[i,j] > thresh_Harris):
                win = pad[i-nms_size:i+nms_size, j-nms_size:j+nms_size]
                if (pad[i,j] == np.max(win)):
                    corners.append((i-nms_size, j-nms_size))

    return corners

def padding(img, padSize):
    img_pad = np.zeros((img.shape[0] + 2*padSize,img.shape[1] + 2*padSize), img.dtype)
    img_pad[padSize:img.shape[0]+padSize, padSize:img.shape[1]+padSize] = img

    return img_pad

def score_ZNCC(patch1, patch2):
    p1 = patch1.flatten()
    p2 = patch2.flatten()

    p1 = p1 - np.mean(p1) 
    p1Norm = np.linalg.norm(p1)

    if p1Norm == 0:
        p1Norm = np.zeros(p1.shape)
    else:
        p1Norm = p1 / p1Norm

    p2 = p2 - np.mean(p2)
    p2Norm = np.linalg.norm(p2)

    if p2Norm == 0:
        p2Norm = np.zeros(p2.shape)
    else:
        p2Norm = p2 / p2Norm

    return np.dot(p1Norm, p2Norm)

def matchKeyPts(img1, img2, patchSize, corners1, corners2, maxScoreThresh):
    match = []

    # Padd images
    i1Pd = padding(img1, patchSize)
    i2Pd = padding(img2, patchSize)
    
    for i, c1 in enumerate(corners1):
        # get first patch
        p1 = i1Pd[c1[0]: c1[0]+patchSize, c1[1]: c1[1]+patchSize]
        bestScore = 0;
        bestPair = (0,0);
        for j, c2 in enumerate(corners2):
            # get second patch
            p2 = i2Pd[c2[0]: c2[0]+patchSize, c2[1]: c2[1]+patchSize]
            zncc = score_ZNCC(p1, p2)
            if (zncc > bestScore):
                bestScore = zncc
                bestPair = (i,j,bestScore)

        if (bestScore > maxScoreThresh):
            match.append(bestPair)

    return match

def hw3(i1, i2):
    c1 = extract_keypts_Harris(i1)
    c2 = extract_keypts_Harris(i2)

    # determine most similar corners
    matches = matchKeyPts(i1, i2, 15, c1, c2, .98)

    draw_matches(i1, i2, c1, c2, matches)

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
    cv2.imshow('HW3', matchingImg)
    cv2.imwrite('cornerMatching.png', matchingImg)


##################
#  main

img1 = cv2.imread('hw3_4/TestingImages/goldengate-02.png')
img2 = cv2.imread('hw3_4/TestingImages/goldengate-03.png')

# convert images to grayscale
img1 = init_image(img1)
img2 = init_image(img2)

hw3(img1, img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
