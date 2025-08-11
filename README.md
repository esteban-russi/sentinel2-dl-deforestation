# Semantic Segmentation for Deforestation Analysis using Sentinel-2 and CNNSs (U-Net) (WIP)

This repository contains the complete code and documentation for an end-to-end deep learning project for forest change analysis in Colombia. The project leverages Sentinel-2 satellite imagery and a U-Net architecture to perform semantic segmentation for land cover classification, with training data sourced from Google's Dynamic World dataset.

## Table of Contents
- [Project Objective](#project-objective)
- [Technical Stack](#technical-stack)
- [Workflow Overview](#workflow-overview)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Objective

The primary goal of this project is to develop an automated pipeline to identify and quantify deforestation over large areas. By training a U-Net model to accurately classify land cover, we can compare predictions from different time periods (e.g., 2021 vs. 2023) to generate forest change maps.

## Technical Stack
- **Cloud Platform:** Google Earth Engine (GEE), Google Cloud Storage (GCS), Google Drive
- **Programming Language:** Python 3.9+
- **Deep Learning Framework:** TensorFlow, Keras
- **Core Libraries:** `earthengine-api`, `geemap`, `numpy`, `matplotlib`, `glob`, `os`
- **Geospatial Data:** Sentinel-2 L2A, Google Dynamic World V1

## Workflow Overview

The project is divided into two main phases:

1.  **Phase 1: Data Engineering (Google Earth Engine)**
    - A cloud-free, median composite of Sentinel-2 imagery is created for the year 2022 over the Colombian Area of Interest.
    - Cloud and shadow masking is applied using the SCL band for data quality.
    - A corresponding label image is generated using the `mode()` of the Dynamic World collection.
    - The stacked feature and label images are exported as 256x256 pixel patches in TFRecord format to Google Drive.

2.  **Phase 2: Model Training & Inference (Local/Cloud VM)**
    - A `tf.data` pipeline efficiently loads and parses the TFRecord files.
    - A U-Net model is built and trained on the prepared dataset.
    - The trained model is evaluated, and its predictions are visualized against the ground truth labels.
    - The final model is used to perform forest change analysis.

## Repository Structure
```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 00-data-exploration.ipynb
├── src/
│   ├── 01-data-export.py
│   ├── 02.0-train-unet.py
│   ├── 02.1-train-att-unet.py
│   ├── 03-change-detection-analysis.py
└── reports/
    └── figures/
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sentinel2-unet-deforestation.git
    cd sentinel2-unet-deforestation
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Authenticate with Google Earth Engine:**
    ```bash
    earthengine authenticate
    ```

## How to Run

1.  **Data Export:** Open and run the `notebooks/01_GEE_Data_Exploration_and_Export.ipynb` notebook. This will guide you through the GEE data processing and trigger the export script (`src/gee_export.py`). Wait for the export task to complete in Google Drive.

2.  **Download Data:** Manually download the exported TFRecord files from your Google Drive `GEE_Colombia_Production_Export` folder and place them in a `data/tfrecords` directory within this project (create it if it doesn't exist).

3.  **Model Training:** Open and run the `notebooks/02_Model_Training_and_Evaluation.ipynb` notebook. This will load the data, train the U-Net model, and generate evaluation results and visualizations.

## Results

This section will showcase the final results of the project, including:
- Model performance metrics (Accuracy, Intersection-over-Union).
- Visual comparisons of model predictions vs. ground truth.
- The final forest change map highlighting areas of deforestation.

## Future Work

- Experiment with different model backbones (e.g., ResNet, EfficientNet) for the U-Net encoder.
- Incorporate additional data sources, such as elevation models (DEM) or SAR data, as input features.
- Deploy the trained model as an inference service on Vertex AI for on-demand analysis of new areas.

## Acknowledgements
- The Copernicus Sentinel-2 program for providing open-access satellite imagery.
- The Google Earth Engine and Dynamic World teams for their invaluable data and platforms.
