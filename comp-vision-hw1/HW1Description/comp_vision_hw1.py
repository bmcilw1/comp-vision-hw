# Brian McIlwain
# Computer Vision
# Homework 1
import cv2

refPt = []
SelectROI = 0
ROI_ONLY = []

def mouseFunc(event, x, y, flags, param):
    #grab references to the global variables
    global refPt, SelectROI    
    if event == cv2.EVENT_LBUTTONDOWN and SelectROI == 1:
        refPt = [(x,y)]
        SelectROI = 2
    elif event == cv2.EVENT_LBUTTONUP and SelectROI == 2:
        refPt.append((x,y))
        SelectROI = 3
        global ROI_ONLY
        ROI_ONLY = myHEQ_YCRCB(img[refPt[0][1]:refPt[1][1],refPt[0][0]:refPt[1][0]])

# Problem 1: Write a function, called myHEQ_RGB(img), to apply histogram equalization directly on a color image.
def myHEQ_RGB(img):
    # Part 1: Split the RGB channels into three, each of which is a grayscale image;
    bImg,gImg,rImg = cv2.split(img)

    # Part 2: Apply equalization on each individual channel;
    bEqu = cv2.equalizeHist(bImg)
    gEqu = cv2.equalizeHist(gImg)
    rEqu = cv2.equalizeHist(rImg)

    # Part 3: Merge the enhanced new channels;
    newImg = cv2.merge([bEqu, gEqu, rEqu])

    # Part 4: Show the result in a window named “HEQ_RGB”
    return newImg

# Problem 2: Write a function, called myHEQ_YCRCB(img), to apply histogram equalization on the Y channel of a color image.
def myHEQ_YCRCB(img):
    # Part 1: Convert the RGB representation to YCrCb representation;
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Part 2: Apply equalization on the Y channel;
    yImg,crImg,cbImg = cv2.split(img_YCrCb)
    yEqu = cv2.equalizeHist(yImg)

    # Part 3: Merge the new Y channel with the other two channels;
    newImgYCrCb = cv2.merge([yEqu, crImg, cbImg])

    # Part 4: Convert the image back to RGB mode.
    newImg = cv2.cvtColor(newImgYCrCb, cv2.COLOR_YCrCb2BGR)

    # Part 5: Show the result in a window named “HEQ_YCRCB”
    return newImg

# Problem 2: Develop a ROI (Region of Interest) Selection Tool and apply Local Equalization:
def myHEQ_ROI(img):
    # Part 1: Implement a mouse-event handling function to allow user to select a rectangular region of Interest (ROI) in the image.
    imgName = 'ROI_SELECT'
    imgCopy = img.copy()
    cv2.imshow(imgName, imgCopy)
    cv2.setMouseCallback(imgName, mouseFunc)
    global SelectROI, refPt
    SelectROI = 1

    while SelectROI != 3:
        cv2.waitKey(0)

    print 'SelectROI == 3'
    # Part 2: Use “myHEQ_YCRCB()” to equalize the ROI region and get an enhanced patch
    newImgROI = myHEQ_YCRCB(ROI_ONLY)

    # Part 3: Update the ROI in the image using this new enhanced patch
    imgCopy[refPt[0][1]:refPt[1][1],refPt[0][0]:refPt[1][0]] = newImgROI

    # Part 5: Show the final result in a window named “HEQ_ROI”
    return imgCopy

# Test your Step 1 and 2 using two under exposed images. And test your Step 3 program using the NikonContest2016Winner and portrait images.
# Save your enhanced images to “HEQ_RGB.png”, “HEQ_YCRCB.png”, and “HEQ_ROI.png” respectively
img = cv2.imread('HW1Description/TestImages/Castle_badexposure.jpg')
#cv2.imshow('DEFAULT', img)

RGB = myHEQ_RGB(img)
cv2.imshow('HEQ_RGB', RGB)
cv2.imwrite('HW1Description/HEQ_RGB.png', RGB)

YCRCB = myHEQ_YCRCB(img)
cv2.imshow('HEQ_YCRCB', YCRCB)
cv2.imwrite('HW1Description/HEQ_YCRCB.png', YCRCB)

ROI = myHEQ_ROI(img)
cv2.imshow('HEQ_ROI', ROI)
cv2.imwrite('HW1Description/HEQ_ROI.png', ROI)

cv2.waitKey(0)

cv2.destroyAllWindows()