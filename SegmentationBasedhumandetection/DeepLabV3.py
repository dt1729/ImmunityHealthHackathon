import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib
import get_dataset_colormap
# Class to load DeepLab model and run inference.
class DeepLab:
    INPUT_TENSOR = 'ImageTensor:0'
    OUTPUT_TENSOR = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    # Creates and loads pretrained Deeplab model
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cant find graph.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.Session(graph=self.graph)
     # Run inference on a single image
    def run(self, image):
        # Args: PIL.image object
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR,
            feed_dict={self.INPUT_TENSOR: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        # Output: RGB image resized from original input image, segmentation map of resized image 
        return resized_image, seg_map

_FROZEN_GRAPH = 'frozen_inference_graph'
#Every time you run the code, a new model will be downloaded. Change the following line to a local path!
model_dir = '/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection'
_TARBALL_NAME = 'deeplab_model.tar.gz'
download_path = os.path.join(model_dir, _TARBALL_NAME)
model = DeepLab(download_path)
cap = cv2.VideoCapture(0)
final = np.zeros((1, 384, 1026, 3))
while True:
    ret, frame = cap.read()
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    # model
    resized_im, seg_map = model.run(pil_im)
    # color of mask
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    
    frame = np.array(pil_im)
    r = seg_image.shape[1] / frame.shape[1]
    dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    color_and_mask = np.hstack((resized, seg_image))
    cv2.imshow('frame', color_and_mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break