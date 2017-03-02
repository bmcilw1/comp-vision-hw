# Brian McIlwain
# Comp vision hw2

import cv2
import numpy as np

def is_Grayscale(img):
    for x in range(0, img.shape[0]):
      for y in range(0, img.shape[1]):
        if img[x, y, 0] != img[x, y, 1] or img[x, y, 1] != img[x, y, 2]:
            return False

    return True

def my_Normalize(img):
    # if already grayscale has no effect
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(src=img, dst=img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return img

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

def my_DerivativesOfGaussian(img, sigma):
    Gsigma = GaussianFilter(sigma)

    Sx = np.matrix([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

    Sy = np.matrix([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

    # Gx = Gsigma * Sx
    Gx = cv2.filter2D(Gsigma, -1, Sx)
    Gy = cv2.filter2D(Gsigma, -1, Sy)
    
    # Ix = img * Gx
    Ix = cv2.filter2D(img, -1, Gx)
    Iy = cv2.filter2D(img, -1, Gy)

    Ixn = np.zeros(shape=(Ix.shape[0], Ix.shape[1]))
    Iyn = np.zeros(shape=(Iy.shape[0], Iy.shape[1]))

    # Normalize for presentation
    Ixn = cv2.normalize(src=Ix, dst=Ixn, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    Iyn = cv2.normalize(src=Iy, dst=Iyn, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    cv2.imshow('Ix_Normalized', Ixn)
    cv2.imshow('Iy_Normalized', Iyn)

    return Ix, Iy

def my_MagAndOrientation(Ix, Iy):
    # Mag and orientation
    M = np.sqrt(Ix*Ix + Iy*Iy)
    O = np.arctan2(Iy,Ix)

    # Normalize mag
    Mn = np.ones(M.shape[0])
    Mn = cv2.normalize(src=M, dst=Mn, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imshow('Mn_Normalized', Mn)

    # Catagorize O
    theta = np.zeros(shape=(M.shape[0],M.shape[1],3))

    for x in range(0, O.shape[0]):
      for y in range(0, O.shape[1]):
          if O[x,y] > -np.pi/8 and O[x,y] <= np.pi/8:
              O[x,y] = 2
              # Blue
              theta[x,y,0] = 255
              theta[x,y,1] = 0
              theta[x,y,2] = 0
          elif O[x,y] > np.pi/8 and O[x,y] <= 3*np.pi/8:
              O[x,y] = 1
              # Green
              theta[x,y,0] = 0
              theta[x,y,1] = 255
              theta[x,y,2] = 0
          elif O[x,y] > -3*np.pi/8 and O[x,y] <= -np.pi/8:
              O[x,y] = 3
              # Gray
              theta[x,y,0] = 128
              theta[x,y,1] = 128
              theta[x,y,2] = 128
          elif (-np.pi/2 < O[x,y] and O[x,y] <= -3*np.pi/8) or (3*np.pi/8 < O[x,y] and O[x,y] <= np.pi/2):
              O[x,y] = 0
              # Red
              theta[x,y,0] = 0
              theta[x,y,1] = 0
              theta[x,y,2] = 255
          else:
              O[x,y] = -1
              # black
              theta[x,y,0] = 0
              theta[x,y,1] = 0
              theta[x,y,2] = 0

    cv2.imshow('O_Catagorized', theta)

    return M, O

def my_NMS(mag, orient, t_low):
    mag_thin = np.zeros(shape=mag.shape)

    for x in range(0, orient.shape[0]):
      for y in range(0, orient.shape[1]):
          if(mag[x,y] >= t_low):
              if(True): # Check if is bigger than its two neighbors in the gradient direction
                  mag_thin[x,y] = mag[x,y]

    cv2.imshow('mag_Thin', mag_thin)

    return mag_thin



def my_Canny(img, sigma, tLow, tHigh):
    img = my_Normalize(img)

    cv2.imshow('input_Normalized', img)

    Ix,Iy = my_DerivativesOfGaussian(img, sigma)

    M,O = my_MagAndOrientation(Ix, Iy)

    mag_thin = my_NMS(M, O, tLow)

    return

img = cv2.imread('hw2/testImages/TestImg1.jpg')

my_Canny(img, .8, .05, 1)

cv2.waitKey(0)
cv2.destroyAllWindows()
