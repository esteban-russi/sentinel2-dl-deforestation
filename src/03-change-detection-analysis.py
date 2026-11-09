"""
03-change-detection-analysis.py

This script performs a post-training change detection analysis using a trained
semantic segmentation model.
"""

try:
    from .config import (
        MODEL_FILE_PATH,
        INFERENCE_T1_FOLDER,
        INFERENCE_T2_FOLDER,
        OUTPUT_FOLDER,
        PATCH_SIZE,
        BANDS,
        TCI_BANDS,
        TCI_INDICES,
        FOREST_CLASS_INDEX,
        PIXEL_RESOLUTION_M,
        CLASS_NAMES_CHANGE,
        CHANGE_MAP_PALETTE,
    )
    from .data import parse_inference_tfrecord, build_inference_dataset
    from .change_detection import (
        reclassify_to_forest_binary,
        compute_change_map,
        quantify_change,
        format_change_report,
        run_change_detection_analysis,
    )
except (ImportError, ValueError):
    from config import (
        MODEL_FILE_PATH,
        INFERENCE_T1_FOLDER,
        INFERENCE_T2_FOLDER,
        OUTPUT_FOLDER,
        PATCH_SIZE,
        BANDS,
        TCI_BANDS,
        TCI_INDICES,
        FOREST_CLASS_INDEX,
        PIXEL_RESOLUTION_M,
        CLASS_NAMES_CHANGE,
        CHANGE_MAP_PALETTE,
    )
    from data import parse_inference_tfrecord, build_inference_dataset
    from change_detection import (
        reclassify_to_forest_binary,
        compute_change_map,
        quantify_change,
        format_change_report,
        run_change_detection_analysis,
    )


def main():
    """Main function to orchestrate the change detection workflow."""
    run_change_detection_analysis()


if __name__ == '__main__':
    main()
