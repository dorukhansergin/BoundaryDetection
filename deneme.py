import os
import pandas as pd
from src import project_utils

IMAGES_DIR = os.path.join("data", "raw", "images")
COLUMN_ANNOTATIONS_DIR = os.path.join("data", "raw", "column_annotations")
CROPPED_DIR = os.path.join("data", "cropped")
FEATURES_DIR = os.path.join("data", "features")
MAX_MARGIN = 33
FIXED_HEIGHT = 300

# # Empty processed and features folders
# try:
#     for dir in [CROPPED_DIR, FEATURES_DIR]:
#         project_utils.recursively_delete_folders(dir)
# except FileNotFoundError as e:
#     print("Directories already don't exist")
#
# project_utils.create_subdirectories(IMAGES_DIR, CROPPED_DIR, FEATURES_DIR)
# wc = project_utils.WindowCropper(IMAGES_DIR, COLUMN_ANNOTATIONS_DIR, CROPPED_DIR, MAX_MARGIN, FIXED_HEIGHT)
# wc.crop_all()

df = project_utils.create_files_dataframe(CROPPED_DIR)
df.to_csv(os.path.join("data_frames", "cropped.csv"))

