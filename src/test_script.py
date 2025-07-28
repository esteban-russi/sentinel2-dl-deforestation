# =============================================================================
# TEST_SCRIPT
# =============================================================================
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- File & Path Configuration ---
DATA_FOLDER = 'tfrecords/'
OUTPUT_FOLDER = 'run_outputs'
FILE_PATTERN = os.path.join(DATA_FOLDER, 'colombia_production_data-*.tfrecord.gz')
LOG_FILE = os.path.join(OUTPUT_FOLDER, 'experiment_log.csv')

# --- Dataset & Image Constants ---
PATCH_SIZE = 256
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
LABEL_BAND = 'label'
TCI_BANDS = ['B4', 'B3', 'B2']
NUM_BANDS = len(BANDS)
TCI_INDICES = [BANDS.index(b) for b in TCI_BANDS]

# --- Label & Visualization Constants ---
NUM_CLASSES = 9
CLASS_NAMES = [
    'Water', 'Trees', 'Grass', 'Flooded Veg', 'Crops',
    'Shrub/Scrub', 'Built Area', 'Bare Ground', 'Snow/Ice'
]

# --- Training Hyperparameters ---
BATCH_SIZE = 1  # Adjusted for CSF3 GPU memory
BUFFER_SIZE = 1000
EPOCHS = 1      # Increased for a full production run
VALIDATION_SPLIT = 0.1

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))


# =============================================================================
print("Loading TFRecord files...")
tfrecord_files = glob.glob(FILE_PATTERN)
if not tfrecord_files:
    raise FileNotFoundError(f"No TFRecord files found matching pattern: {FILE_PATTERN}")
print(f"Found {len(tfrecord_files)} TFRecord files.")

raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')

print("Counting total records...")
# Count the number of records in the dataset
dataset_size = raw_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
print(f"Total records in dataset: {dataset_size}")

# =============================================================================

print("Test Script finished successfully.")