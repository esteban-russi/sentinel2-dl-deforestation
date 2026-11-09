"""
config.py

Centralized configuration, paths, hyperparameters, and domain constants
for the Sentinel-2 deforestation and land cover segmentation pipeline.
"""

import os

# =============================================================================
# --- 1. File & Path Configuration ---
# =============================================================================
# All data and output folders live under the repository root, whatever the
# working directory the scripts are started from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(REPO_ROOT, 'tfrecords')
OUTPUT_FOLDER = os.path.join(REPO_ROOT, 'run_outputs')
FILE_PATTERN = os.path.join(DATA_FOLDER, 'colombia_production_data-*.tfrecord.gz')
LOG_FILE = os.path.join(OUTPUT_FOLDER, 'experiment_log.csv')

# Change detection inference paths
MODEL_FILE_PATH = os.path.join(REPO_ROOT, 'best_models', 'best_model_Attention_U-Net.keras')
INFERENCE_T1_FOLDER = os.path.join(REPO_ROOT, 'inference_tfrecords_2021')
INFERENCE_T2_FOLDER = os.path.join(REPO_ROOT, 'inference_tfrecords_2023')

# =============================================================================
# --- 2. Dataset & Image Constants ---
# =============================================================================
PATCH_SIZE = 256
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
LABEL_BAND = 'label'
TCI_BANDS = ['B4', 'B3', 'B2']
NUM_BANDS = len(BANDS)
TCI_INDICES = [BANDS.index(b) for b in TCI_BANDS]
EXPORT_SCALE = 10
PIXEL_RESOLUTION_M = 10

# =============================================================================
# --- 3. Class Definitions & Color Palettes ---
# =============================================================================
# Original Dynamic World classes:
# 0:water, 1:trees, 2:grass, 3:flooded_veg, 4:crops, 5:shrub, 6:built, 7:bare, 8:snow
# Our 4-class mapping:
# 0: Other (0, 3, 4, 6, 7, 8)
# 1: Trees (1)
# 2: Grass (2)
# 3: Shrub/Scrub (5)
NUM_CLASSES = 4
CLASS_NAMES_4 = ['Other', 'Trees', 'Grass', 'Shrub/Scrub']
FOREST_CLASS_INDEX = 1

# Binary mapping for final evaluation and change detection
BINARY_CLASS_NAMES = ['Non-Forest', 'Forest']

# Change detection classes & palette
CLASS_NAMES_CHANGE = ['Deforestation', 'No Change', 'Reforestation']
CHANGE_MAP_PALETTE = ['#d7191c', '#ffffbf', '#2c7bb6']  # Red -> Yellow -> Blue

# 4-class visualization palette (Other -> Grey, Trees -> Dark Green, Grass -> Light Green, Shrub -> Olive/Yellow-Green)
VIS_PALETTE_4_CLASS = ['#9B9B9B', '#006400', '#88b053', '#dfc35a']

# =============================================================================
# --- 4. Training Hyperparameters ---
# =============================================================================
BATCH_SIZE = 5
BUFFER_SIZE = 250
EPOCHS = 50
VALIDATION_SPLIT = 0.1
SPLIT_SEED = 42  # Fixes which patches end up in the validation split
VALIDATION_STEPS_PER_EPOCH = 15
# The baseline U-Net runs a fixed number of training steps per epoch, sized for
# 2051 training patches. The Attention U-Net lets Keras infer the epoch length.
# If the training split has fewer patches than this, Keras runs out of data.
UNET_TRAINING_PATCHES = 2051
UNET_STEPS_PER_EPOCH = UNET_TRAINING_PATCHES // BATCH_SIZE

RUN_NOTES = (
    f"full run; BUFFER_SIZE = {BUFFER_SIZE}; BATCH_SIZE = {BATCH_SIZE}; "
    f"EPOCHS = {EPOCHS}; kept VAL_STEPS_PER_EPOCH = {VALIDATION_STEPS_PER_EPOCH}"
)

# =============================================================================
# --- 5. Earth Engine Export Configuration ---
# =============================================================================
ROI_TRAINING_BBOX = (-74.88851, 1.722, -73.656, 2.712)
YEAR_TRAINING = 2022
DRIVE_FOLDER_TRAINING = 'GEE_Colombia_Training_Export'

ROI_CHANGE_DETECTION_BBOX = (-72.8, 2.1, -72.2, 2.7)
YEAR_T1_INFERENCE = 2021
YEAR_T2_INFERENCE = 2023
DRIVE_FOLDER_INFERENCE = 'GEE_Colombia_ChangeDetection_Export'
