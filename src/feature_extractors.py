import os

import numpy as np
import scipy.io as io
from PIL import Image
from skimage.feature import hog


class WCFeatureExtractor:
    def __init__(self, column_annotations_dir, images_dir, wc):
        self.column_annotations_dir = column_annotations_dir
        self.images_dir = images_dir
        self.wc = wc

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
        img = np.array(Image.open(os.path.join(self.images_dir, filename + '.jpg')))
        boundary_locations = io.loadmat(
            os.path.join(self.column_annotations_dir, filename + '.mat'))['cols'].astype(np.uint).reshape(-1)
        pos, neg = self.wc.crop_all(img, boundary_locations)
        return filename, np.vstack(list(map(lambda crop: hog(crop, block_norm='L2-Hys'), pos+neg))), \
               np.array([1]*len(pos)+[0]*len(neg))


class WindowCropper:
    """
    A tool to crop positive and negative boundary images from a given shelf image
    """
    def __init__(self, margin):
        # make sure the margin is odd
        assert margin % 2 == 1, "margin ({}) is not odd, please choose an odd margin ".format(margin)
        self.margin = margin

    def crop_all(self, image, boundary_locations):
        return self.crop_positives(image, boundary_locations), self.crop_negatives(image, boundary_locations)

    def crop_single(self, image, location):
        """
        :param image: numpy array, the whole shelf image to be cropped
        :param location: integer, the index of the center of crop
        :return:
        An image of width margin in which location is centered
        """
        arm_length = int((self.margin - 1) / 2)
        if location-arm_length >= 0 & location-arm_length < image.shape[1]:
            return image[:, location - arm_length:location + arm_length]
        elif location-arm_length >= image.shape[1]:
            return image[:, image.shape[1]-self.margin:image.shape[1]]
        else:
            return image[:, 0:self.margin]

    def crop_positives(self, image, boundary_locations):
        return list(map(lambda location: self.crop_single(image, location), boundary_locations))

    def crop_negatives(self, image, boundary_locations):
        return [self.crop_single(image, int((boundary_locations[i] + boundary_locations[i+1])/2))
         for i in range(len(boundary_locations) - 1)]