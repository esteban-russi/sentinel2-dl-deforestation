"""
03-change-detection-analysis.py

This script performs a post-training change detection analysis using a trained
semantic segmentation model.
"""

from change_detection import run_change_detection_analysis


def main():
    """Main function to orchestrate the change detection workflow."""
    run_change_detection_analysis()


if __name__ == '__main__':
    main()
