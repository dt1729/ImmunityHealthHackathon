#Some imports
import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
# import get_dataset_colormap

#Download URLs of the pre-trained Xception model
_MODEL_URLS = {
    'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}

Config = collections.namedtuple('Config', 'model_url, model_dir')

def get_config(model_name, model_dir):
    return Config(_MODEL_URLS[model_name], model_dir)

# config_widget = interactive(get_config, model_name=_MODEL_URLS.keys(), model_dir='')
# display.display(config_widget)
model_dir = '/Users/dt/Desktop/CodesTemp/ImmunityHealth/ImmunityHealthHackathon/SegmentationBasedhumandetection'
_TARBALL_NAME = 'deeplab_model.tar.gz'


tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model to %s, this might take a while...' % download_path)
urllib.request.urlretrieve(_MODEL_URLS['xception_coco_voctrainval'], download_path)
print('download completed!')