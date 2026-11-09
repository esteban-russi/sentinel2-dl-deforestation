"""
change_detection.py

Core change detection computation, spatial metric quantification,
and full analysis orchestration.
"""

import os
import time
from datetime import datetime
import numpy as np

try:
    from .config import (
        MODEL_FILE_PATH,
        INFERENCE_T1_FOLDER,
        INFERENCE_T2_FOLDER,
        OUTPUT_FOLDER,
        FOREST_CLASS_INDEX,
        PIXEL_RESOLUTION_M,
    )
    from .visualization import save_change_visualizations
except (ImportError, ValueError):
    from config import (
        MODEL_FILE_PATH,
        INFERENCE_T1_FOLDER,
        INFERENCE_T2_FOLDER,
        OUTPUT_FOLDER,
        FOREST_CLASS_INDEX,
        PIXEL_RESOLUTION_M,
    )
    from visualization import save_change_visualizations


def reclassify_to_forest_binary(predicted_labels, forest_class_index=FOREST_CLASS_INDEX):
    """
    Reclassifies multiclass predictions to binary Forest (1) vs. Non-Forest (0).
    """
    return np.where(predicted_labels == forest_class_index, 1, 0)


def compute_change_map(forest_map_t1, forest_map_t2):
    """
    Calculates the change map: T2 - T1.

    Values:
      -1: Deforestation (Forest in T1 -> Non-Forest in T2)
       0: No Change
       1: Reforestation (Non-Forest in T1 -> Forest in T2)
    """
    return forest_map_t2 - forest_map_t1


def quantify_change(change_map, pixel_resolution_m=PIXEL_RESOLUTION_M):
    """
    Quantifies the total pixels and area in km² for deforestation,
    reforestation, total area, and net change.
    """
    pixels_deforested = int(np.sum(change_map == -1))
    pixels_reforested = int(np.sum(change_map == 1))
    total_pixels = int(change_map.size)

    sq_meters_per_pixel = pixel_resolution_m ** 2
    area_deforested_sq_km = (pixels_deforested * sq_meters_per_pixel) / 1_000_000
    area_reforested_sq_km = (pixels_reforested * sq_meters_per_pixel) / 1_000_000
    total_area_sq_km = (total_pixels * sq_meters_per_pixel) / 1_000_000
    net_change_sq_km = area_reforested_sq_km - area_deforested_sq_km

    return {
        'pixels_deforested': pixels_deforested,
        'pixels_reforested': pixels_reforested,
        'total_pixels': total_pixels,
        'area_deforested_sq_km': area_deforested_sq_km,
        'area_reforested_sq_km': area_reforested_sq_km,
        'total_area_sq_km': total_area_sq_km,
        'net_change_sq_km': net_change_sq_km,
    }


def format_change_report(metrics):
    """Formats change detection metrics into a human-readable text report."""
    return (
        f"==========================================================\n"
        f"      Change Detection Analysis Report\n"
        f"==========================================================\n"
        f"Analysis Period: T1 (Before) vs. T2 (After)\n"
        f"Total Analyzed Area: {metrics['total_area_sq_km']:,.2f} km²\n"
        f"----------------------------------------------------------\n"
        f"Deforestation (Forest -> Non-Forest):\n"
        f"  - Pixels Changed: {metrics['pixels_deforested']:,}\n"
        f"  - Area Changed:   {metrics['area_deforested_sq_km']:,.2f} km²\n"
        f"\nReforestation (Non-Forest -> Forest):\n"
        f"  - Pixels Changed: {metrics['pixels_reforested']:,}\n"
        f"  - Area Changed:   {metrics['area_reforested_sq_km']:,.2f} km²\n"
        f"==========================================================\n"
    )


def run_change_detection_analysis(
    model_file_path=MODEL_FILE_PATH,
    inference_t1_folder=INFERENCE_T1_FOLDER,
    inference_t2_folder=INFERENCE_T2_FOLDER,
    output_folder=OUTPUT_FOLDER,
    batch_size=32,
    num_samples_to_viz=10,
):
    """
    Executes the full post-training change detection analysis workflow.
    """
    import tensorflow as tf
    try:
        from .data import build_inference_dataset
    except (ImportError, ValueError):
        from data import build_inference_dataset

    start_time = time.time()
    print("="*60 + "\n--- Starting Change Detection Analysis Workflow ---\n" + "="*60)

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_folder = os.path.join(output_folder, f'change_analysis_{run_timestamp}')
    os.makedirs(run_output_folder, exist_ok=True)
    print(f"All analysis outputs will be saved in: {run_output_folder}")

    # 1. Load the Trained Model
    print(f"\n--- Loading trained model from: {model_file_path} ---")
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    model = tf.keras.models.load_model(model_file_path)
    model.summary()

    # 2. Build Inference Datasets
    print("\n--- Building inference data pipelines ---")
    dataset_t1, raw_images_t1 = build_inference_dataset(inference_t1_folder, batch_size=batch_size)
    dataset_t2, raw_images_t2 = build_inference_dataset(inference_t2_folder, batch_size=batch_size)

    # 3. Run Inference
    print("\n--- Running inference on T1 (Before) dataset ---")
    predictions_t1 = model.predict(dataset_t1, verbose=1)
    predicted_labels_t1 = np.argmax(predictions_t1, axis=-1)

    print("\n--- Running inference on T2 (After) dataset ---")
    predictions_t2 = model.predict(dataset_t2, verbose=1)
    predicted_labels_t2 = np.argmax(predictions_t2, axis=-1)

    if predicted_labels_t1.shape != predicted_labels_t2.shape:
        print(
            "WARNING: T1 and T2 datasets have a different number of patches. "
            "Quantification will proceed on the minimum of the two."
        )
        min_patches = min(predicted_labels_t1.shape[0], predicted_labels_t2.shape[0])
        predicted_labels_t1 = predicted_labels_t1[:min_patches]
        predicted_labels_t2 = predicted_labels_t2[:min_patches]

    print(f"Inference complete. Total patches processed: {predicted_labels_t1.shape[0]}")

    # 4. Post-Process for Change Analysis
    print("\n--- Post-processing maps for change analysis ---")
    forest_map_t1 = reclassify_to_forest_binary(predicted_labels_t1)
    forest_map_t2 = reclassify_to_forest_binary(predicted_labels_t2)
    change_map = compute_change_map(forest_map_t1, forest_map_t2)

    # 5. Quantify Change
    print("\n--- Quantifying Land Cover Change ---")
    metrics = quantify_change(change_map)
    report_text = format_change_report(metrics)
    print(report_text)

    report_path = os.path.join(run_output_folder, 'change_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    # 6. Visualize Change
    print("\n--- Generating change map visualizations ---")
    viz_path = os.path.join(run_output_folder, 'change_visualizations.png')
    save_change_visualizations(
        change_map,
        raw_images_t1,
        raw_images_t2,
        viz_path,
        num_samples_to_viz=num_samples_to_viz,
    )

    end_time = time.time()
    print("\n" + "="*60 + "\n--- Change Detection Workflow Complete ---\n" + "="*60)
    print(f"Total analysis time: {(end_time - start_time) / 60:.2f} minutes")
    return metrics
