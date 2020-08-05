import numpy as np
import cv2
import os
from PIL import Image
from DeepLabV3 import DeepLab
import get_dataset_colormap
import json 



cap = cv2.VideoCapture(0)
model_dir = '/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection'
_TARBALL_NAME = 'deeplab_model.tar.gz'
_FROZEN_GRAPH = 'frozen_inference_graph'
download_path = os.path.join(model_dir, _TARBALL_NAME)
model = DeepLab(download_path)


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


def referenceBackground():
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
            resized_im, seg_map = model.run(pil_im)
            # color of mask
            seg_image = get_dataset_colormap.label_to_color_image(
                seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
            # cv2.imshow('background',seg_map)
            unique_labels,label_counts = np.unique(seg_map, return_counts=True)
            in_this_pic = LABEL_NAMES[unique_labels]
            grouped_label_counts  = dict(zip(in_this_pic,label_counts))
            # saving semantic data as json
            save_json(grouped_label_counts,'reference')
            # loadcheck 
            open_json('reference.txt')
            cv2.imwrite('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection/referenceBackgroundSegmented.jpg', seg_image)
            break
    cv2.destroyAllWindows()
    return seg_image,seg_map



def save_json(grouped_label_counts,filename):
    data = grouped_label_counts
    with open(filename + '.txt', 'w') as outfile:
        json.dump(data, outfile)

def open_json(filename):
    data = json.load(filename)
    print(data);

def postureCorrection():
    ref_Seg = cv2.imread('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection/referenceBackgroundSegmented.jpg')
    while(cap.isOpened()):
        pass
    pass



if __name__ == "__main__":
    reference = input("have you taken reference before? Y or N")
    if reference == "N":
        seg_img,seg_map =  referenceBackground()
    else:
        segnums = open_json('reference.txt')
    cap.release()
    



    
    



