# Brian McIlwain
# Comp vision hw2

import cv2

def myNormalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.normalize(src=img, dst=img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img

img = cv2.imread('hw2/testImages/ChambordCastle.jpg')

img = myNormalize(img)
cv2.imshow('myNormalize', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
