import numpy as np
import cv2
import os
from PIL import Image
from DeepLabV3 import DeepLab
import get_dataset_colormap
import csv
import time

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

model_dir = '/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection'
_TARBALL_NAME = 'deeplab_model.tar.gz'
_FROZEN_GRAPH = 'frozen_inference_graph'

def referenceBackground(cap):
    while (True):
        ret1, frame1 = cap.read()
        frame1RGB = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        print('Capture reference image first')
        print('press p to capture background')
        cv2.imshow('background',frame1RGB)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.destroyAllWindows()
            print('in p')
            pil_im = Image.fromarray(frame1RGB)
            # model
            model_time = time.time()
            resized_im, seg_map = model.run(pil_im)
            # color of mask
            seg_image = get_dataset_colormap.label_to_color_image(
                seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
            model_time = (time.time() - model_time)
            # cv2.imshow('background',seg_map)
            unique_labels,label_counts = np.unique(seg_map, return_counts=True)
            in_this_pic = LABEL_NAMES[unique_labels]
            grouped_label_counts  = dict(zip(in_this_pic,label_counts))
            # np.savetxt('reference.csv',grouped_label_counts,delimiter=',')
            # saving semantic data as json
            save_csv(grouped_label_counts,'reference')
            # loadcheck 
            # open_json('reference.txt')
            cv2.imwrite('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection/referenceBackgroundSegmented.jpg', seg_image)
            break
    print('done capturing reference!')
    cv2.destroyAllWindows()
    return grouped_label_counts, model_time


def save_csv(grouped_label_counts,filename):
    with open(filename+'.csv', 'w') as f:
        for key in grouped_label_counts.keys():
            f.write("%s, %s\n" % (key, grouped_label_counts[key]))

def open_csv(filename):
    with open(filename + '.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        grouped_label_counts = {}
        for row in reader:
            grouped_label_counts[row[0]] = int(row[1])

    return grouped_label_counts

def postureCorrection(reference_time):
    cap = cv2.VideoCapture(0)
    grouped_label_counts_reference = open_csv('reference')
    prev_time = time.time()
    while(cap.isOpened()):
        if (time.time() - prev_time) > reference_time: 
            ret1, frame1 = cap.read()
            print("in here")
            frame1RGB = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(frame1RGB)
            resized_im, seg_map = model.run(pil_im)
            seg_image = get_dataset_colormap.label_to_color_image(
                    seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
            unique_labels,label_counts = np.unique(seg_map, return_counts=True)
            in_this_pic = LABEL_NAMES[unique_labels]
            grouped_label_counts  = dict(zip(in_this_pic,label_counts))
            if grouped_label_counts['person'] > grouped_label_counts_reference['person']:
                print("Please adjust your posture and move a little backwards")
            elif grouped_label_counts['person'] < grouped_label_counts_reference['person']:
                print("Please adjust your posture and move a little foward")
            prev_time = time.time()



if __name__ == "__main__":
    reference = input("have you taken reference before? Y or N")
    if reference == "N" or reference == "n":
        cap = cv2.VideoCapture(0)
        download_path = os.path.join(model_dir, _TARBALL_NAME)
        model = DeepLab(download_path)
        grouped_label_counts, model_time =  referenceBackground(cap)
        if 'cap' in locals():
            cap.release()

    # starting with real-time posture check based on pixels covered
    postureCorrection(model_time)
    if 'cap' in locals():
        cap.release()
    



    
    



