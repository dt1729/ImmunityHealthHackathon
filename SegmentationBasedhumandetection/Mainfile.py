import numpy as np
import cv2
import os
from PIL import Image
from DeepLabV3 import DeepLab

cap = cv2.VideoCapture(0)
model_dir = '/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection'
_TARBALL_NAME = 'deeplab_model.tar.gz'
_FROZEN_GRAPH = 'frozen_inference_graph'
download_path = os.path.join(model_dir, _TARBALL_NAME)
model = DeepLab(download_path)
def referenceBackground():
    while (True):
        ret1, frame1 = cap.read()
        frame1RGB = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        print('Capture reference image first')
        print('press p to capture background')
        cv2.imshow('background',frame1RGB)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print('in p')
            pil_im = Image.fromarray(cv2_im)
            # model
            resized_im, seg_map = model.run(pil_im)
            # color of mask
            seg_image = get_dataset_colormap.label_to_color_image(
                seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
            cv2.imwrite('/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection/referenceBackgroundSegmented.jpg', seg_image)
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    return seg_image





    
    
cap.release()


