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
        # fgmask = fgbg.apply(grayfiltered)

        # Display the resulting frame
        edges = cv2.Canny(grayfiltered,cannyLowerThresh,cannyUpperThresh)
        cv2.imshow('frame',edges)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print('in p')
            cv2.imwrite('/Users/dt/Desktop/CodesTemp/ImmunityHealth/referenceImg.jpg', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    frame = cap.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/referenceImg.jpg',0)
    cap.release()
    cv2.destroyAllWindows()
    return frame



if __name__ == "__main__":
    kernelSize = 4
    cannyLowerThresh = 50
    cannyUpperThresh = 70
    referenceEdge = captureImageandVideo(kernelSize,cannyLowerThresh,cannyUpperThresh)
    # temporarily for cropped image
    referenceEdge = cap.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/referenceImg.jpg',0)