# Brian McIlwain
# Comp vision hw3 and hw4
 
import cv2
import numpy as np

def init_image(img):
    # Convert to grayscale
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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
    # Padd image by padSize with zeros
    img_pad = np.zeros((img.shape[0] + 2*padSize,img.shape[1] + 2*padSize), img.dtype)
    img_pad[padSize:img.shape[0]+padSize, padSize:img.shape[1]+padSize] = img

    return img_pad

def score_ZNCC(patch1, patch2):
    # Allow us to use numpy built ins
    p1 = patch1.flatten()
    p2 = patch2.flatten()

    # ZNCC as in slides
    p1 = p1 - np.mean(p1) 
    p1Norm = np.linalg.norm(p1)
    p1Norm = np.zeros(p1.shape) if p1Norm == 0 else p1 / p1Norm

    p2 = p2 - np.mean(p2)
    p2Norm = np.linalg.norm(p2)
    p2Norm = np.zeros(p2.shape) if p2Norm == 0 else p2 / p2Norm

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
    # Get corners 
    c1 = extract_keypts_Harris(i1)
    c2 = extract_keypts_Harris(i2)

    # determine most similar corners
    matches = matchKeyPts(i1, i2, 15, c1, c2, .98)

    # show it
    draw_matches(i1, i2, c1, c2, matches)

    return c1, c2, matches

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
    cv2.imwrite('hw3_4/cornerMatching.png', matchingImg)

def compute_Homography(corners1, corners2, matches):
    A = np.zeros((2*len(matches), 8))
    b = np.zeros(2*len(matches))

    # Initalize A and B
    for i in range(len(matches)):
        # set (x,y) (x', y')
        y = corners1[matches[i][0]][0]
        x = corners1[matches[i][0]][1]
        yp = corners1[matches[i][1]][0]
        xp = corners1[matches[i][1]][1]

        # Set b as instructed in handout
        b[2*i] = xp
        b[2*i+1] = yp

        # Set up A as in demo
        A[2*i,0] = x
        A[2*i,1] = y
        A[2*i,2] = 1
        A[2*i,6] = -x * xp
        A[2*i,7] = -y * yp
        A[2*i+1,3] = x
        A[2*i+1,4] = y
        A[2*i+1,5] = 1
        A[2*i+1,6] = -x * xp
        A[2*i+1,7] = -y * yp

    # Compute H
    H = np.linalg.lstsq(A, b)[0]

    # Append constant 1, fix dimensions
    H = np.append(H, [1])
    H = np.reshape(H,(3,3))
    print H
    return H

def apply_transform(T, x, y):
    # Compute (u,v,w)
    u = T[0,0]*x + T[0,1]*y + T[0,2]
    v = T[1,0]*x + T[1,1]*y + T[1,2]
    w = T[2,0]*x + T[2,1]*y + T[2,2]

    # Use (u,v,w) to return (x',y') transform
    xT = u / w
    yT = v / w

    return xT, yT

def compute_StitchDimension(img1, img2, H):

    return

def hw4(i1, i2, c1, c2, matches):
    # Get H
    H = compute_Homography(c1, c2, matches)

    # Get shape
    compute_StitchDimension(i1, i2, H)

    return

##################
#  main

img1 = cv2.imread('hw3_4/TestingImages/goldengate-02.png')
img2 = cv2.imread('hw3_4/TestingImages/goldengate-03.png')

# convert images to grayscale
img1 = init_image(img1)
img2 = init_image(img2)

# Get matching points
c1, c2, matches = hw3(img1, img2)

# Stich images together
hw4(img1, img2, c1, c2, matches)

cv2.waitKey(0)
cv2.destroyAllWindows()
