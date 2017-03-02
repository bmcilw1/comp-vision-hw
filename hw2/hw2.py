# Brian McIlwain
# Comp vision hw2

import cv2
import numpy as np

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

def my_Normalize(img):
    img = img.copy()

    # Is Grayscale?
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(src=img, dst=img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    return img

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
    Ixn = my_Normalize(Ix)
    Iyn = my_Normalize(Iy)

    cv2.imshow('Ix_Normalized', Ixn)
    cv2.imwrite('hw2/Ix_Normalized.jpg', Ixn)
    cv2.imshow('Iy_Normalized', Iyn)
    cv2.imwrite('hw2/Iy_Normalized.jpg', Iyn)

    return Ix, Iy

def my_MagAndOrientation(Ix, Iy, t_low):
    # Mag and orientation
    M = np.sqrt(Ix*Ix + Iy*Iy)
    O = np.arctan2(Iy,Ix)

    # Normalize mag
    Mn = np.ones(M.shape[0])
    Mn = my_Normalize(M)

    cv2.imshow('Mag_Normalized', Mn)
    cv2.imwrite('hw2/Mag_Normalized.jpg', Mn)

    # Catagorize O
    theta = np.zeros(shape=(M.shape[0],M.shape[1],3))

    for x in range(0, O.shape[0]):
      for y in range(0, O.shape[1]):
          if M[x,y] < t_low:
              O[x,y] = -1
          elif (-3*np.pi/8 < O[x,y] and O[x,y] <= -np.pi/8) or (5*np.pi/8 < O[x,y] and O[x,y] <= 7*np.pi/8):
              O[x,y] = 1
              # Green
              theta[x,y,0] = 0
              theta[x,y,1] = 255
              theta[x,y,2] = 0
          elif (np.pi/8 < O[x,y] and O[x,y] <= 3*np.pi/8) or (-7*np.pi/8 < O[x,y] and O[x,y] <= -5*np.pi/8):
              O[x,y] = 3
              # Gray
              theta[x,y,0] = 128
              theta[x,y,1] = 128
              theta[x,y,2] = 128
          elif (-5*np.pi/8 < O[x,y] and O[x,y] <= -3*np.pi/8) or (3*np.pi/8 < O[x,y] and O[x,y] <= 5*np.pi/8):
              O[x,y] = 2
              # Blue
              theta[x,y,0] = 255
              theta[x,y,1] = 0
              theta[x,y,2] = 0
          else:
              O[x,y] = 0
              # Red
              theta[x,y,0] = 0
              theta[x,y,1] = 0
              theta[x,y,2] = 255

    cv2.imshow('O_Catagorized', theta)
    cv2.imwrite('hw2/O_Catagorized.jpg', theta)

    return M, O

def my_NMS(mag, orient, t_low):
    mag_thin = np.zeros(shape=mag.shape)

    for x in range(0, orient.shape[0]):
      for y in range(0, orient.shape[1]):
          if(mag[x,y] >= t_low):
              # Check if is bigger than its two neighbors in the gradient direction
              if(orient[x,y] == 0 and (mag[x,y-1]   <  mag[x,y] and mag[x,y+1]   <=  mag[x,y]) or
                 orient[x,y] == 1 and (mag[x+1,y-1] <  mag[x,y] and mag[x-1,y+1] <=  mag[x,y]) or
                 orient[x,y] == 2 and (mag[x-1,y]   <  mag[x,y] and mag[x+1,y]   <=  mag[x,y]) or
                 orient[x,y] == 3 and (mag[x+1,y+1] <  mag[x,y] and mag[x-1,y-1] <=  mag[x,y])):
                      mag_thin[x,y] = mag[x,y]

    cv2.imshow('Mag_thin', mag_thin)
    cv2.imwrite('hw2/Mag_thin.jpg', mag_thin)

    return mag_thin

def my_linking(mag_thin, orient, tLow, tHigh):
    result_binary = np.zeros(shape=mag_thin.shape)

    for x in range(0, mag_thin.shape[0]-1):
      for y in range(0, mag_thin.shape[1]-1):
          if(mag_thin[x][y] >= tHigh):
              if (tLow <= mag_thin[x+1][y]):
                  mag_thin[x+1][y] = 1
              elif (tLow <= mag_thin[x+1][y+1]):
                  mag_thin[x+1][y+1] = 1
              elif (tLow <= mag_thin[x][y+1]):
                  mag_thin[x][y+1] = 1
              elif (tLow <= mag_thin[x-1][y+1]):
                  mag_thin[x-1][y+1] = 1

    for x in range(mag_thin.shape[0], 1):
      for y in range(mag_thin.shape[1], 1):
          if(mag_thin[x][y] >= tHigh):
              if (tLow <= mag_thin[x-1][y]):
                  mag_thin[x-1][y] = 1
              elif (tLow <= mag_thin[x-1][y-1]):
                  mag_thin[x-1][y-1] = 1
              elif (tLow <= mag_thin[x][y-1]):
                  mag_thin[x][y-1] = 1
              elif (tLow <= mag_thin[x+1][y-1]):
                  mag_thin[x+1][y-1] = 1

    cv2.imshow('Result_binary', result_binary)
    cv2.imwrite('hw2/Result_binary.jpg', result_binary)

    return result_binary

def my_Canny(img, sigma, tLow, tHigh):
    img = my_Normalize(img)

    cv2.imshow('input_Normalized', img)
    cv2.imwrite('hw2/input_Normalized.jpg', img)

    Ix,Iy = my_DerivativesOfGaussian(img, sigma)

    M,O = my_MagAndOrientation(Ix, Iy, tLow)

    mag_thin = my_NMS(M, O, tLow)

    result_binary = my_linking(mag_thin, O, tLow, tHigh)

    return

##################
#  main

img = cv2.imread('hw2/testImages/TestImg1.jpg')

my_Canny(img, .8, .05, .1)

cv2.waitKey(0)
cv2.destroyAllWindows()
