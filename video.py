import numpy as np
import cv2

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
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # binarizing the image
    ret, thresh = cv2.threshold(frame, 127, 255, 0)
    return thresh

def detectContours(binarizedImage):
    _, contours, _ = cv2.findContours(binarizedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == "__main__":
    kernelSize = 4
    cannyLowerThresh = 50
    cannyUpperThresh = 70
    # referenceEdge = captureImageandVideo(kernelSize,cannyLowerThresh,cannyUpperThresh)
    # temporarily for cropped image
    referenceEdge = cv2.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/referenceImg.png',0)
    contour = detectContours(referenceEdge)
    img = cv2.drawContours(referenceEdge, contour, -1, (0,255,0), 3)
    print(np.shape(img))
    cv2.imshow('frame',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
