import os
import shutil

import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import rescale
from scipy.io import loadmat
SUBDIR_NAMES = ["pos", "neg", "pos_aug", "neg_aug"]

# Create relevant cropped image directories
def create_subdirectories(shelf_images_dir, cropped_images_dir, features_dir):
    """
    Prepares directories for cropped images and respective features
    """
    # shelf_names holds the image names like ['shelf1', 'shelf2', ... ]
    for dir in [cropped_images_dir, features_dir]:
        shelf_names = map(lambda filename: filename.split(".")[0], os.listdir(shelf_images_dir))
        if not os.path.exists(dir):
            os.mkdir(dir)
        for shelf_name in shelf_names:
            shelf_dir_path = os.path.join(dir, shelf_name)
            os.mkdir(shelf_dir_path)
            for subdir_name in SUBDIR_NAMES:
                os.mkdir(os.path.join(shelf_dir_path, subdir_name))


class WindowCropper:
    """
    A tool to crop positive and negative boundary images from a given shelf image
    """
    def __init__(self, images_dir, column_annotations_dir, cropped_dir, margin, fixed_height):
        # make sure the margin is odd
        assert margin % 2 == 1, "margin ({}) is not odd, please choose an odd margin ".format(margin)
        self.margin = margin
        self.arm_length = int((self.margin - 1) / 2)
        self.images_dir = images_dir
        self.column_annotations_dir = column_annotations_dir
        self.cropped_dir = cropped_dir
        self.fixed_height = fixed_height

    def crop_all(self):
        for filename in map(lambda filename: filename.split(".")[0], os.listdir(self.images_dir)):
            img = io.imread(os.path.join(self.images_dir, filename + '.jpg'))
            scaling_ratio = self.fixed_height / img.shape[0]
            img = rescale(img, scaling_ratio)
            boundary_locations = loadmat(os.path.join(self.column_annotations_dir, filename + '.mat'))['cols']\
                .astype(np.uint).reshape(-1)
            boundary_locations = scaling_ratio * boundary_locations
            self.crop_positives(img, boundary_locations, os.path.join(self.cropped_dir, filename, "pos"))
            self.crop_negatives(img, boundary_locations, os.path.join(self.cropped_dir, filename, "neg"))

    def crop_single(self, image, location):
        """
        :param image: numpy array, the whole shelf image to be cropped
        :param location: integer, the index of the center of crop
        :return:
        An image of width margin in which location is centered
        """
        if ((location - self.arm_length) >= 0) & ((location + self.arm_length) < image.shape[1]):
            crop = image[:, location - self.arm_length:location + self.arm_length]
        elif (location + self.arm_length) >= image.shape[1]:
            crop = image[:, (-self.margin+1):]
        else:
            crop = image[:, :self.margin-1]
        return crop

    def crop_positives(self, image, boundary_locations, path):
        for i, location in enumerate(boundary_locations):
            io.imsave(os.path.join(path, str(i)+".jpg"), self.crop_single(image, int(location)))

    def crop_negatives(self, image, boundary_locations, path):
        for i in range(len(boundary_locations) - 1):
            io.imsave(os.path.join(path, str(i)+".jpg"),
                      self.crop_single(image, int((boundary_locations[i] + boundary_locations[i+1])/2)))


# Recursively delete all cropped images
def recursively_delete_folders(path):
    shutil.rmtree(path)


# Create cropped images pandas dataframe
def create_files_dataframe(cropped_images_dir):
    df = pd.DataFrame(None, columns=["shelf", "path", "aug", "label"])
    for shelf in os.listdir(cropped_images_dir):
        shelf_path = os.path.join(cropped_images_dir, shelf)
        for subdir in os.listdir(shelf_path):
            subdir_path = os.path.join(shelf_path, subdir)
            for image in os.listdir(subdir_path):
                dict_to_append = {"shelf": shelf,
                                  "path": os.path.join(subdir_path, image),
                                  "aug": int("aug" in os.path.basename(subdir_path)),
                                  "label": int("pos" in os.path.basename(subdir_path))}
                df = df.append(dict_to_append, ignore_index=True)
    return df

# Create features pandas dataframe