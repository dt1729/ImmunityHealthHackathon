import numpy as np
import cv2
from skimage import data, img_as_float
from skimage.segmentation import chan_vese


cap = cv2.VideoCapture(0)
def referenceBackground():
    while (cap.isOpened()):
        ret1, frame1 = cap.read()
        frame1 = frame1[:,160:1119,:]
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        print('we have to capture your background first')
        print('press p to capture background')
        cv2.imshow('background',gray1)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print('in p')
            cv2.imwrite('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/referenceBackgroundImg.jpg', gray1)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    gray1 = cv2.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/referenceBackgroundImg.jpg')
    return gray1

def captureImageandVideo(kernelsize = 3,cannyLowerThresh = 50,cannyUpperThresh = 70):
    cap = cv2.VideoCapture(0)
    # background removal mask from opencv
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = frame[:,160:1119,:]
        # print(np.shape(frame))
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #filtering the grayscale image 
        kernel = np.ones((kernelsize,kernelsize),np.float32)/(kernelsize**2)
        # kernel[:,0] = np.zeros((1,num),np.float32)
        print(kernelsize)
        grayfiltered = cv2.filter2D(gray,-1,kernel)
        # removing background of the video using background substractor

        # Canny edge detector to generate a binary image
        edges = cv2.Canny(grayfiltered,cannyLowerThresh,cannyUpperThresh)
        # Display the resulting frame
        cv2.imshow('frame',edges)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print('in p')
            cv2.imwrite('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/referenceImg.jpg', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    frame = cap.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/referenceImg.jpg',0)
    cap.release()
    cv2.destroyAllWindows()
    return frame

def binarizingImage(image):
    # making sure that the cropped image is in binary
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 150, 200, cv2.THRESH_BINARY)
    return ret,thresh

def backgroundRemovalNaive(backgroundImg):
    cap = cv2.VideoCapture(0)
    prevframe = backgroundImg
    while True:
        ret,realtimeImage = cap.read()
        realtimeImage = realtimeImage[:,160:1119,:]
        realtimeImage = cv2.cvtColor(realtimeImage, cv2.COLOR_BGR2GRAY)
        humanfromRealTime = realtimeImage - prevframe
        ret,binarizedHuman = binarizingImage(humanfromRealTime)
        cv2.imshow('human',binarizedHuman)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prevframe = realtimeImage
    cap.release()
    cv2.destroyAllWindows()

def backgroundRemovalChanVese(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    while True:
        ret,realtimeImage = cap.read()
        realtimeImage = realtimeImage[:,160:1119,:]
        realtimeImage = cv2.cvtColor(realtimeImage, cv2.COLOR_BGR2GRAY)
        humanfromRealTime = chan_vese(realtimeImage, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
               dt=0.5, init_level_set="checkerboard", extended_output=True)
        cv2.imshow('human',humanfromRealTime)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return cv
def DetectandDrawContours(binarizedImage,draw = 0):
    _, contours, _ = cv2.findContours(binarizedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if draw:
        print('Drawing Contours')
        cv2.drawContours(binarizedImage, contours, -1, (255,0,0), 1)
    return binarizedImage,contours

if __name__ == "__main__":
    kernelSize = 4
    cannyLowerThresh = 50
    cannyUpperThresh = 70
    # referenceEdge = captureImageandVideo(kernelSize,cannyLowerThresh,cannyUpperThresh)
    # temporarily for cropped image
    referenceBackgroundimg = referenceBackground()
    referenceEdge = cv2.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/referenceImg.png',0)
    humanImg = backgroundRemovalChanVese(referenceBackgroundimg)
    _,binarizedImage = binarizingImage(referenceEdge)
    binarizedImage,contours = DetectandDrawContours(binarizedImage,0)
    temp = binarizedImage
    for i in range(np.shape(contours)[0]):
        contours[i] = contours[i].reshape(-1,2)
        for (x, y) in contours[i]:
            # cv2.circle(temp, (x, y), 1, (255, 0, 0), 1)
            pass
    cv2.imshow('frame',binarizedImage)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
