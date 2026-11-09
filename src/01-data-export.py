# -*- coding: utf-8 -*-
"""
01-data-export.py

This master script handles all data export tasks from Google Earth Engine for the
deforestation analysis project. It can be configured to export either:
1. The full training dataset (Sentinel-2 features + Dynamic World labels).
2. Inference-only datasets (Sentinel-2 features only) for specific years for
   the change detection analysis.
"""

import ee

try:
    from .config import (
        BANDS as S2_BANDS,
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
    from .gee_export import (
        initialize_ee,
        mask_s2_clouds,
        create_s2_composite,
        run_training_data_export,
        run_inference_data_export,
    )
except (ImportError, ValueError):
    from config import (
        BANDS as S2_BANDS,
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
    from gee_export import (
        initialize_ee,
        mask_s2_clouds,
        create_s2_composite,
        run_training_data_export,
        run_inference_data_export,
    )


# Geometry definitions for GEE
ROI_TRAINING = ee.Geometry.BBox(*ROI_TRAINING_BBOX)
ROI_CHANGE_DETECTION = ee.Geometry.BBox(*ROI_CHANGE_DETECTION_BBOX)


if __name__ == '__main__':
    initialize_ee()

    # Generate the dataset for training and change detection:
    run_training_data_export(roi=ROI_TRAINING)
    run_inference_data_export(YEAR_T1_INFERENCE, roi=ROI_CHANGE_DETECTION)
    run_inference_data_export(YEAR_T2_INFERENCE, roi=ROI_CHANGE_DETECTION)

    print("\n\n============================================================")
    print("All selected export tasks have been submitted.")
    print("--> Go to the 'Tasks' tab in the GEE Code Editor to monitor progress.")
    print("============================================================")
