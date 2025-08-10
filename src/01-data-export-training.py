# -*- coding: utf-8 -*-
"""
master_gee_export.py

This master script handles all data export tasks from Google Earth Engine for the
deforestation analysis project. It can be configured to export either:
1. The full training dataset (Sentinel-2 features + Dynamic World labels).
2. Inference-only datasets (Sentinel-2 features only) for specific years for
   the change detection analysis.

This unified approach ensures that the exact same data processing methodology
(e.g., cloud masking) is applied to both training and inference data, which is
critical for accurate and reliable model performance.

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

# =============================================================================
# 0. IMPORTS AND INITIALIZATION
# =============================================================================
import ee

try:
    ee.Initialize(project="deforestationsentinel2")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="deforestationsentinel2")

print("Google Earth Engine Initialized.")


# =============================================================================
# 1. MASTER CONFIGURATION
# =============================================================================

# --- General Export Settings ---
S2_BANDS: list = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
PATCH_SIZE: int = 256
EXPORT_SCALE: int = 10

# --- Configuration for TRAINING Data Export ---
# The large ROI used to generate a diverse training dataset.
ROI_TRAINING = ee.Geometry.BBox(-74.88851, 1.722, -73.656, 2.712)
YEAR_TRAINING: int = 2022
DRIVE_FOLDER_TRAINING: str = 'GEE_Colombia_Training_Export'

# --- Configuration for CHANGE DETECTION (Inference) Export ---
# The specific ROI for the final analysis.
ROI_CHANGE_DETECTION = ee.Geometry.BBox(-72.8, 2.1, -72.2, 2.7)
YEAR_T1_INFERENCE: int = 2021  # "Before" year
YEAR_T2_INFERENCE: int = 2023  # "After" year
DRIVE_FOLDER_INFERENCE: str = 'GEE_Colombia_ChangeDetection_Export'


# =============================================================================
# 2. SHARED DATA PREPARATION FUNCTIONS
# =============================================================================

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds and shadows in a Sentinel-2 SR image using the SCL band."""
    scl = image.select('SCL')
    # Keep high-quality land, water, and snow/ice pixels.
    good_quality = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(good_quality)


def create_s2_composite(roi: ee.Geometry, year: int) -> ee.Image:
    """
    Creates a normalized, cloud-free median composite of Sentinel-2 imagery for a given ROI and year.
    """
    print(f"Preparing Sentinel-2 composite for {year}...")
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(roi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25)))
    
    s2_masked = s2_collection.map(mask_s2_clouds)
    s2_composite = s2_masked.select(S2_BANDS).median()
    feature_image = s2_composite.divide(3000).toFloat()
    
    return feature_image


# =============================================================================
# 3. EXPORT WORKFLOWS
# =============================================================================

def run_training_data_export() -> None:
    """
    Prepares and exports the full training dataset (features + labels).
    """
    print("\n" + "="*50 + "\n--- STARTING TRAINING DATA EXPORT ---" + "\n" + "="*50)
    
    # 1. Create the feature image composite
    feature_image = create_s2_composite(ROI_TRAINING, YEAR_TRAINING)
    
    # 2. Create the corresponding label image
    print(f"Preparing Dynamic World label image for {YEAR_TRAINING}...")
    start_date = f'{YEAR_TRAINING}-01-01'
    end_date = f'{YEAR_TRAINING}-12-31'
    
    dw_collection = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                     .filterBounds(ROI_TRAINING)
                     .filterDate(start_date, end_date))
    dw_image = dw_collection.select('label').mode()
    label_image = dw_image.toByte()
    
    # 3. Stack features and labels
    stacked_image = feature_image.addBands(label_image)
    print("Data preparation complete. Image stack created.")

    # 4. Configure and start the export task
    task_description = f'TrainingExport_{YEAR_TRAINING}'
    file_prefix = f'training_data_{YEAR_TRAINING}'
    
    print(f"Submitting export task: '{task_description}'...")
    task = ee.batch.Export.image.toDrive(
        image=stacked_image,
        description=task_description,
        folder=DRIVE_FOLDER_TRAINING,
        fileNamePrefix=file_prefix,
        region=ROI_TRAINING,
        scale=EXPORT_SCALE,
        fileFormat='TFRecord',
        formatOptions={'patchDimensions': [PATCH_SIZE, PATCH_SIZE], 'compressed': True},
        maxPixels=1e13
    )
    task.start()
    print(f"SUCCESS: Task '{task_description}' has been started.")
    print("="*50)


def run_inference_data_export(year: int) -> None:
    """
    Prepares and exports an inference-only dataset (features only) for a given year.
    """
    print("\n" + "="*50 + f"\n--- STARTING INFERENCE DATA EXPORT for {year} ---" + "\n" + "="*50)

    # 1. Create the feature image composite
    feature_image = create_s2_composite(ROI_CHANGE_DETECTION, year)
    print("Data preparation complete.")

    # 2. Configure and start the export task
    task_description = f'InferenceExport_{year}'
    file_prefix = f'inference_data_{year}'
    
    print(f"Submitting export task: '{task_description}'...")
    task = ee.batch.Export.image.toDrive(
        image=feature_image,
        description=task_description,
        folder=DRIVE_FOLDER_INFERENCE,
        fileNamePrefix=file_prefix,
        region=ROI_CHANGE_DETECTION,
        scale=EXPORT_SCALE,
        fileFormat='TFRecord',
        formatOptions={'patchDimensions': [PATCH_SIZE, PATCH_SIZE], 'compressed': True},
        maxPixels=1e13
    )
    task.start()
    print(f"SUCCESS: Task '{task_description}' has been started.")
    print("="*50)


# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    # --- Control which exports to run by commenting/uncommenting these lines ---
    
    # To generate the dataset for training your model:
    # run_training_data_export()
    
    # To generate the "before" and "after" datasets for change detection:
    run_training_data_export()
    run_inference_data_export(YEAR_T1_INFERENCE)
    run_inference_data_export(YEAR_T2_INFERENCE)
    
    print("\n\n============================================================")
    print("All selected export tasks have been submitted.")
    print("--> Go to the 'Tasks' tab in the GEE Code Editor to monitor progress.")
    print("============================================================")