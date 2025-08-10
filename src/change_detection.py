"""
change_detection_analysis.py

This script performs a post-training change detection analysis using a trained
semantic segmentation model.

Workflow:
1. Loads the best-trained model (e.g., Attention U-Net).
2. Defines file paths for the "before" (T1) and "after" (T2) inference datasets.
3. Creates efficient tf.data pipelines to load the inference data.
4. Runs batch predictions (inference) on both datasets to generate 4-class land cover maps.
5. Post-processes the 4-class maps into binary "Forest" vs. "Non-Forest" maps.
6. Calculates the change map by subtracting the T1 map from the T2 map.
7. Quantifies the total area of deforestation and reforestation in square kilometers.
8. Generates and saves visualizations of sample patches showing land cover change.
9. Saves all outputs to a timestamped folder for reproducibility.
"""

# =============================================================================
# --- 0. Preamble and Imports ---
# =============================================================================
import os
import time
import glob
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Configure Matplotlib for non-GUI backend
plt.switch_backend('Agg')

# =============================================================================
# --- 1. Configuration ---
# =============================================================================
# --- Path Configuration ---
# Point this to your best saved model from the training run
MODEL_FILE_PATH = 'run_outputs/20250809_210609/best_model_Attention_U-Net.keras'
# Point these to the folders containing your downloaded inference TFRecords
INFERENCE_T1_FOLDER = 'inference_tfrecords_2021/' # "Before" data
INFERENCE_T2_FOLDER = 'inference_tfrecords_2023/' # "After" data
OUTPUT_FOLDER = 'run_outputs'

# --- Model & Data Constants (Must match training) ---
PATCH_SIZE = 256
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
TCI_BANDS = ['B4', 'B3', 'B2']
TCI_INDICES = [BANDS.index(b) for b in TCI_BANDS]

# --- Analysis Constants ---
# Class mapping: The index of the "Trees" class in your 4-class model output
FOREST_CLASS_INDEX = 1
# Pixel resolution in meters for area calculation
PIXEL_RESOLUTION_M = 10

# --- Visualization Constants ---
CLASS_NAMES_CHANGE = ['Deforestation', 'No Change', 'Reforestation']
# Use a clear, diverging colormap for the change map
CHANGE_MAP_PALETTE = ['#d7191c', '#ffffbf', '#2c7bb6'] # Red -> Yellow -> Blue

# =============================================================================
# --- 2. Data Pipeline for Inference ---
# =============================================================================
def parse_inference_tfrecord(example_proto):
    """Parses a TFRecord for inference (features only, no labels)."""
    feature_description = {band: tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32) for band in BANDS}
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    features_list = [tf.reshape(example[band], [PATCH_SIZE, PATCH_SIZE]) for band in BANDS]
    image = tf.stack(features_list, axis=-1)
    return image

def build_inference_dataset(folder_path, batch_size=32):
    """Builds a tf.data pipeline for a given set of inference TFRecord files."""
    file_pattern = os.path.join(folder_path, '*.tfrecord.gz')
    tfrecord_files = glob.glob(file_pattern)
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in: {folder_path}")
    
    print(f"Found {len(tfrecord_files)} files in {folder_path}.")
    
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    dataset = dataset.map(parse_inference_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    # NOTE: Do NOT shuffle inference data. We need to process patches in order.
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # We also need the raw image data for visualization later
    raw_image_dataset = dataset.unbatch()
    
    return dataset, raw_image_dataset

# =============================================================================
# --- 3. Main Analysis Workflow ---
# =============================================================================
def main():
    """Main function to orchestrate the change detection workflow."""
    start_time = time.time()
    
    print("="*60 + "\n--- Starting Change Detection Analysis Workflow ---\n" + "="*60)
    
    # --- Create a unique output directory for this analysis run ---
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_folder = os.path.join(OUTPUT_FOLDER, f'change_analysis_{run_timestamp}')
    os.makedirs(run_output_folder, exist_ok=True)
    print(f"All analysis outputs will be saved in: {run_output_folder}")

    # --- 1. Load the Trained Model ---
    print(f"\n--- Loading trained model from: {MODEL_FILE_PATH} ---")
    if not os.path.exists(MODEL_FILE_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE_PATH}")
    model = tf.keras.models.load_model(MODEL_FILE_PATH)
    model.summary()
    
    # --- 2. Build Inference Datasets ---
    print("\n--- Building inference data pipelines ---")
    dataset_t1, raw_images_t1 = build_inference_dataset(INFERENCE_T1_FOLDER)
    dataset_t2, raw_images_t2 = build_inference_dataset(INFERENCE_T2_FOLDER)
    
    # --- 3. Run Inference ---
    print("\n--- Running inference on T1 (Before) dataset ---")
    predictions_t1 = model.predict(dataset_t1, verbose=1)
    predicted_labels_t1 = np.argmax(predictions_t1, axis=-1)
    
    print("\n--- Running inference on T2 (After) dataset ---")
    predictions_t2 = model.predict(dataset_t2, verbose=1)
    predicted_labels_t2 = np.argmax(predictions_t2, axis=-1)
    
    if predicted_labels_t1.shape != predicted_labels_t2.shape:
        print("WARNING: T1 and T2 datasets have a different number of patches. "
              "Quantification will proceed on the minimum of the two.")
        min_patches = min(predicted_labels_t1.shape[0], predicted_labels_t2.shape[0])
        predicted_labels_t1 = predicted_labels_t1[:min_patches]
        predicted_labels_t2 = predicted_labels_t2[:min_patches]

    print(f"Inference complete. Total patches processed: {predicted_labels_t1.shape[0]}")
    
    # --- 4. Post-Process for Change Analysis ---
    print("\n--- Post-processing maps for change analysis ---")
    
    # Reclassify 4-class maps to binary Forest (1) vs. Non-Forest (0)
    forest_map_t1 = np.where(predicted_labels_t1 == FOREST_CLASS_INDEX, 1, 0)
    forest_map_t2 = np.where(predicted_labels_t2 == FOREST_CLASS_INDEX, 1, 0)
    
    # Calculate the change map: T2 - T1
    # Result: -1 (Deforestation), 0 (No Change), 1 (Reforestation)
    change_map = forest_map_t2 - forest_map_t1
    
    # --- 5. Quantify Change ---
    print("\n--- Quantifying Land Cover Change ---")
    pixels_deforested = np.sum(change_map == -1)
    pixels_reforested = np.sum(change_map == 1)
    total_pixels = change_map.size
    
    sq_meters_per_pixel = PIXEL_RESOLUTION_M ** 2
    
    area_deforested_sq_km = (pixels_deforested * sq_meters_per_pixel) / 1_000_000
    area_reforested_sq_km = (pixels_reforested * sq_meters_per_pixel) / 1_000_000
    total_area_sq_km = (total_pixels * sq_meters_per_pixel) / 1_000_000
    
    report_text = (
        f"==========================================================\n"
        f"      Change Detection Analysis Report\n"
        f"==========================================================\n"
        f"Analysis Period: T1 (Before) vs. T2 (After)\n"
        f"Total Analyzed Area: {total_area_sq_km:,.2f} km²\n"
        f"----------------------------------------------------------\n"
        f"Deforestation (Forest -> Non-Forest):\n"
        f"  - Pixels Changed: {pixels_deforested:,}\n"
        f"  - Area Changed:   {area_deforested_sq_km:,.2f} km²\n"
        f"\nReforestation (Non-Forest -> Forest):\n"
        f"  - Pixels Changed: {pixels_reforested:,}\n"
        f"  - Area Changed:   {area_reforested_sq_km:,.2f} km²\n"
        f"==========================================================\n"
    )
    print(report_text)
    
    # Save the report to a text file
    with open(os.path.join(run_output_folder, 'change_analysis_report.txt'), 'w') as f:
        f.write(report_text)
        
    # --- 6. Visualize Change ---
    print("\n--- Generating change map visualizations ---")
    num_samples_to_viz = 10
    
    # Create colormap and patches for the legend
    cmap_change = ListedColormap(CHANGE_MAP_PALETTE)
    patches = [Patch(color=color, label=name) for color, name in zip(CHANGE_MAP_PALETTE, CLASS_NAMES_CHANGE)]

    fig, axes = plt.subplots(num_samples_to_viz, 3, figsize=(15, 5 * num_samples_to_viz))
    fig.suptitle('Sample Patches of Land Cover Change', fontsize=16, y=1.0)
    
    # Create iterators to get corresponding raw images
    raw_t1_iter = raw_images_t1.as_numpy_iterator()
    raw_t2_iter = raw_images_t2.as_numpy_iterator()

    for i in range(num_samples_to_viz):
        try:
            # Get the TCI images for T1 and T2
            image_t1_numpy = next(raw_t1_iter)
            image_t2_numpy = next(raw_t2_iter)
            
            tci_t1 = np.clip(image_t1_numpy[..., TCI_INDICES], 0, 1) ** (1/1.8)
            tci_t2 = np.clip(image_t2_numpy[..., TCI_INDICES], 0, 1) ** (1/1.8)
            
            # Get the corresponding change patch
            change_patch = change_map[i]
            
            # Plot T1 Image
            axes[i, 0].imshow(tci_t1)
            axes[i, 0].set_title(f"Patch #{i} - T1 (Before)")
            axes[i, 0].axis('off')

            # Plot T2 Image
            axes[i, 1].imshow(tci_t2)
            axes[i, 1].set_title(f"Patch #{i} - T2 (After)")
            axes[i, 1].axis('off')
            
            # Plot Change Map
            im = axes[i, 2].imshow(change_patch, cmap=cmap_change, vmin=-1, vmax=1)
            axes[i, 2].set_title(f"Patch #{i} - Change Map")
            axes[i, 2].axis('off')
            
        except StopIteration:
            # Hide unused subplots if we have fewer samples than requested
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')

    # Add a single shared legend to the figure
    fig.legend(handles=patches, bbox_to_anchor=(1.01, 0.9), loc='upper left', title="Change Types")
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    
    viz_path = os.path.join(run_output_folder, 'change_visualizations.png')
    plt.savefig(viz_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualizations saved to {viz_path}")
    
    end_time = time.time()
    print("\n" + "="*60 + "\n--- Change Detection Workflow Complete ---\n" + "="*60)
    print(f"Total analysis time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    main()