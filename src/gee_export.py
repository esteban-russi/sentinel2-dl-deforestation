"""
gee_export.py

Google Earth Engine helper functions for composite creation and patch export.
Initialization is handled cleanly on-demand rather than immediately on import.
"""

import ee

from config import (
    BANDS,
    PATCH_SIZE,
    EXPORT_SCALE,
    ROI_TRAINING_BBOX,
    YEAR_TRAINING,
    DRIVE_FOLDER_TRAINING,
    ROI_CHANGE_DETECTION_BBOX,
    YEAR_T1_INFERENCE,
    YEAR_T2_INFERENCE,
    DRIVE_FOLDER_INFERENCE,
)


def initialize_ee(project="deforestationsentinel2"):
    """Initializes Google Earth Engine, authenticating if necessary."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print("Google Earth Engine Initialized.")


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds and shadows in a Sentinel-2 SR image using the SCL band."""
    scl = image.select('SCL')
    # Keep high-quality land, water, and snow/ice pixels.
    good_quality = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(good_quality)


def create_s2_composite(
    roi: ee.Geometry,
    year: int,
    bands: list = BANDS,
    max_cloud_percentage: int = 25,
) -> ee.Image:
    """
    Creates a normalized, cloud-free median composite of Sentinel-2 imagery for a given ROI and year.
    """
    print(f"Preparing Sentinel-2 composite for {year}...")
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    s2_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percentage))
    )

    s2_masked = s2_collection.map(mask_s2_clouds)
    s2_composite = s2_masked.select(bands).median()
    feature_image = s2_composite.divide(3000).toFloat()

    return feature_image


def run_training_data_export(
    roi: ee.Geometry = None,
    year: int = YEAR_TRAINING,
    drive_folder: str = DRIVE_FOLDER_TRAINING,
    patch_size: int = PATCH_SIZE,
    scale: int = EXPORT_SCALE,
) -> None:
    """Prepares and exports the full training dataset (features + Dynamic World labels)."""
    if roi is None:
        roi = ee.Geometry.BBox(*ROI_TRAINING_BBOX)

    print("\n" + "="*50 + "\n--- STARTING TRAINING DATA EXPORT ---" + "\n" + "="*50)

    # 1. Create the feature image composite
    feature_image = create_s2_composite(roi, year)

    # 2. Create the corresponding label image
    print(f"Preparing Dynamic World label image for {year}...")
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    dw_collection = (
        ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
    )
    dw_image = dw_collection.select('label').mode()
    label_image = dw_image.toByte()

    # 3. Stack features and labels
    stacked_image = feature_image.addBands(label_image)
    print("Data preparation complete. Image stack created.")

    # 4. Configure and start the export task
    task_description = f'TrainingExport_{year}'
    file_prefix = f'training_data_{year}'

    print(f"Submitting export task: '{task_description}'...")
    task = ee.batch.Export.image.toDrive(
        image=stacked_image,
        description=task_description,
        folder=drive_folder,
        fileNamePrefix=file_prefix,
        region=roi,
        scale=scale,
        fileFormat='TFRecord',
        formatOptions={'patchDimensions': [patch_size, patch_size], 'compressed': True},
        maxPixels=1e13,
    )
    task.start()
    print(f"SUCCESS: Task '{task_description}' has been started.")
    print("="*50)


def run_inference_data_export(
    year: int,
    roi: ee.Geometry = None,
    drive_folder: str = DRIVE_FOLDER_INFERENCE,
    patch_size: int = PATCH_SIZE,
    scale: int = EXPORT_SCALE,
) -> None:
    """Prepares and exports an inference-only dataset (features only) for a given year."""
    if roi is None:
        roi = ee.Geometry.BBox(*ROI_CHANGE_DETECTION_BBOX)

    print("\n" + "="*50 + f"\n--- STARTING INFERENCE DATA EXPORT for {year} ---" + "\n" + "="*50)

    # 1. Create the feature image composite
    feature_image = create_s2_composite(roi, year)
    print("Data preparation complete.")

    # 2. Configure and start the export task
    task_description = f'InferenceExport_{year}'
    file_prefix = f'inference_data_{year}'

    print(f"Submitting export task: '{task_description}'...")
    task = ee.batch.Export.image.toDrive(
        image=feature_image,
        description=task_description,
        folder=drive_folder,
        fileNamePrefix=file_prefix,
        region=roi,
        scale=scale,
        fileFormat='TFRecord',
        formatOptions={'patchDimensions': [patch_size, patch_size], 'compressed': True},
        maxPixels=1e13,
    )
    task.start()
    print(f"SUCCESS: Task '{task_description}' has been started.")
    print("="*50)
