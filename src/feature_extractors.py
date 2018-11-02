import os

import numpy as np
import scipy.io as io
from PIL import Image
from skimage.feature import hog
from skimage.transform import rescale

MIN_HEIGHT = 300





class WCFeatureExtractor:
    def __init__(self, margin):
        self.images_dir = None
        self.column_annotations_dir = None
        self.wc = WindowCropper(margin)

    def set_directories(self, column_annotations_dir, images_dir):
        self.column_annotations_dir = column_annotations_dir
        self.images_dir = images_dir

    def get_features_and_labels(self, filename):
        """
        Generic extraction function for any feature extractor
        Must return the following:
            filename: the name of the image file where features are extracted from
            feat_mat: a numpy matrix with d feature dimensions as columns and n rows fow n cropped windows
            label_vec: a numpy array with n labels, one each per positive (1) and negative crops (0)
        """
        raise NotImplementedError


class HOGExtractor(WCFeatureExtractor):
    def get_features_and_labels(self, filename):
        img = np.array(Image.open(os.path.join(self.images_dir, filename + '.jpg')).convert('L'))
        scaling_ratio = MIN_HEIGHT/img.shape[0]
        img = rescale(img, MIN_HEIGHT/img.shape[0])
        boundary_locations = io.loadmat(
            os.path.join(self.column_annotations_dir, filename + '.mat'))['cols'].astype(np.uint).reshape(-1)
        boundary_locations = scaling_ratio * boundary_locations
        pos, neg = self.wc.crop_all(img, boundary_locations)

        return filename, np.vstack(list(map(lambda crop: hog(crop, block_norm='L2-Hys', cells_per_block=(3, 1)),
                                            pos+neg))), np.array([1]*len(pos)+[0]*len(neg))

