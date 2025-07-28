"""
train_models.py

This script performs an end-to-end training and evaluation workflow for
semantic segmentation models on satellite imagery. It is designed to be
run in a non-interactive environment like an HPC cluster (e.g., CSF3).

Workflow:
1.  Configures constants and hyperparameters.
2.  Builds a high-performance tf.data pipeline from TFRecord files.
3.  Defines, compiles, and trains one or more segmentation models (e.g., U-Net).
4.  Evaluates the best performing models on the full validation set.
5.  Saves results: model weights, training history plots, evaluation reports,
    and prediction visualizations.
6.  Logs a summary of the experiment run to a CSV file for tracking.
"""

# =============================================================================
# --- 0. Preamble and Imports ---
# =============================================================================
import os
import time
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configure Matplotlib for non-GUI backend (crucial for HPC)
plt.switch_backend('Agg')

# =============================================================================
# --- 1. Configuration & Constants ---
# =============================================================================
# --- Run-specific Notes ---
RUN_NOTES = "U-Net baseline training on full Colombian dataset on CSF3."

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

# =============================================================================
# --- 2. Data Pipeline ---
# =============================================================================
def parse_tfrecord(example_proto):
    """Parses a single TFRecord example into image and one-hot label tensors."""
    feature_description = {band: tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32) for band in BANDS}
    feature_description[LABEL_BAND] = tf.io.FixedLenFeature([], tf.string)
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    features_list = [tf.reshape(example[band], [PATCH_SIZE, PATCH_SIZE]) for band in BANDS]
    image = tf.stack(features_list, axis=-1)
    
    label_decoded = tf.io.decode_raw(example[LABEL_BAND], tf.uint8)
    label = tf.reshape(label_decoded, [PATCH_SIZE, PATCH_SIZE])
    label_one_hot = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
    
    return image, label_one_hot

def build_dataset(file_pattern, batch_size, buffer_size, val_split):
    """Builds, splits, and batches the full dataset for training and validation."""
    tfrecord_files = glob.glob(file_pattern)
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found matching pattern: {file_pattern}")
    print(f"Found {len(tfrecord_files)} TFRecord files.")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    
    print("Counting total records...")
    dataset_size = raw_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

    # Create the full data pipeline
    dataset = raw_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    
    # Split into training and validation sets
    train_size = int((1 - val_split) * dataset_size)
    train_size = 10
    
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    
    print(f"Splitting into {train_size} training samples and {dataset_size - train_size} validation samples.")
    
    # Apply batching and prefetching
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("Data pipeline built and split successfully.")
    
    return train_dataset, validation_dataset, train_size, dataset_size - train_size

# =============================================================================
# --- 3. Model Definitions ---
# =============================================================================
def build_unet_model(input_shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS), num_classes=NUM_CLASSES):
    """Builds a U-Net model with dropout for regularization."""
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    b = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    b = layers.Dropout(0.3)(b)
    b = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(b)
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u6 = layers.concatenate([u6, c3])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c2])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c1])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c8)
    model = Model(inputs=[inputs], outputs=[outputs], name="U-Net")
    return model

# Add other model definitions (e.g., ResU-Net) here if you want to compare them.

# =============================================================================
# --- 4. Training and Evaluation Functions ---
# =============================================================================

def compile_and_train(model, train_data, val_data, epochs, output_folder):
    """Compiles, trains, and returns a model and its history."""
    model_name = model.name
    print(f"\n--- Starting Workflow for Model: {model_name} ---")
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=list(range(NUM_CLASSES)))])
    
    print(f"Model '{model_name}' compiled successfully.")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_folder, f'best_model_{model_name}.keras'),
                                           save_best_only=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=2 # Use verbose=2 for cleaner logs in a batch job
    )
    print(f"--- Finished Training: {model_name} ---")
    return history

def evaluate_and_report_metrics(models_dict, validation_dataset, output_folder):
    """Evaluates models, prints reports, saves plots, and returns accuracies."""
    print("\n" + "="*50 + "\n--- GENERATING FINAL EVALUATION METRICS ---\n" + "="*50)
    
    all_true_labels, all_pred_labels_dict = [], {name: [] for name in models_dict.keys()}
    
    for image_batch, label_batch in validation_dataset:
        true_labels = np.argmax(label_batch.numpy(), axis=-1).flatten()
        all_true_labels.extend(true_labels)
        for name, model in models_dict.items():
            pred_batch = model.predict(image_batch, verbose=0)
            pred_labels = np.argmax(pred_batch, axis=-1).flatten()
            all_pred_labels_dict[name].extend(pred_labels)
            
    model_accuracies = {}
    for name, pred_labels in all_pred_labels_dict.items():
        print("\n" + "-"*20 + f" REPORT FOR MODEL: {name} " + "-"*20)
        report = classification_report(all_true_labels, pred_labels, target_names=CLASS_NAMES, zero_division=0)
        print("\nClassification Report:\n", report)
        
        accuracy = accuracy_score(all_true_labels, pred_labels)
        print(f"Overall Pixel Accuracy: {accuracy:.4f}")
        model_accuracies[name] = accuracy
        
        cm = confusion_matrix(all_true_labels, pred_labels, labels=np.arange(NUM_CLASSES))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix - {name}', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_folder, f'confusion_matrix_{name}.png'), bbox_inches='tight')
        plt.close() # Close the plot to free up memory

    return model_accuracies

def save_visualizations(models_dict, val_data, output_folder, num_samples=5):
    """Saves model prediction visualizations to files."""
    print("\n--- Saving Final Prediction Visualizations ---")
    if not models_dict: return

    image_batch, label_batch = next(iter(val_data))
    true_labels = np.argmax(label_batch.numpy(), axis=-1)
    predictions = {name: np.argmax(model.predict(image_batch, verbose=0), axis=-1) for name, model in models_dict.items()}
    
    num_models = len(models_dict)
    fig_cols = 2 + num_models
    
    plt.figure(figsize=(5 * fig_cols, 5 * num_samples))
    for i in range(min(num_samples, image_batch.shape[0])):
        plt.subplot(num_samples, fig_cols, i * fig_cols + 1)
        tci_image = image_batch.numpy()[i][..., TCI_INDICES]
        plt.imshow(np.clip(tci_image, 0, 1) ** (1 / 1.8))
        plt.title(f"Input #{i+1}"); plt.axis('off')

        plt.subplot(num_samples, fig_cols, i * fig_cols + 2)
        plt.imshow(true_labels[i], cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
        plt.title("Ground Truth"); plt.axis('off')
        
        for j, (name, pred_labels) in enumerate(predictions.items()):
            plt.subplot(num_samples, fig_cols, i * fig_cols + 3 + j)
            plt.imshow(pred_labels[i], cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
            plt.title(f"{name} Prediction"); plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'prediction_visualizations.png'), bbox_inches='tight')
    plt.close()

def save_training_histories(history_dict, output_folder):
    """Saves plots of training histories to a file."""
    print("\n--- Saving Training History Plots ---")
    plt.figure(figsize=(20, 7))
    plt.suptitle('Model Performance Comparison', fontsize=16)

    # Plot Loss
    plt.subplot(1, 2, 1)
    for name, history in history_dict.items():
        plt.plot(history.history['loss'], label=f'{name} Train Loss', linestyle='--')
        plt.plot(history.history['val_loss'], label=f'{name} Val Loss', linestyle='-')
    plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    # Plot IoU
    plt.subplot(1, 2, 2)
    for name, history in history_dict.items():
        iou_key = next((k for k in history.history if 'one_hot_io_u' in k and 'val_' not in k), None)
        val_iou_key = next((k for k in history.history if 'val_one_hot_io_u' in k), None)
        if iou_key and val_iou_key:
            plt.plot(history.history[iou_key], label=f'{name} Train IoU', linestyle='--')
            plt.plot(history.history[val_iou_key], label=f'{name} Val IoU', linestyle='-')
    plt.title('Training & Validation Mean IoU'); plt.xlabel('Epoch'); plt.ylabel('Mean IoU')
    plt.legend(); plt.grid(True)

    plt.savefig(os.path.join(output_folder, 'training_histories.png'), bbox_inches='tight')
    plt.close()

def log_experiment(log_file_path, run_data):
    """Appends a summary of the run to a CSV log file."""
    new_log_df = pd.DataFrame([run_data])
    write_header = not os.path.exists(log_file_path)
    new_log_df.to_csv(log_file_path, mode='a', header=write_header, index=False)
    print("\n" + "="*50 + f"\nSUCCESS: Run results logged to '{log_file_path}'\n" + "="*50)

# =============================================================================
# --- 5. Main Execution Block ---
# =============================================================================
def main():
    """Main function to orchestrate the entire workflow."""
    start_time = time.time()
    
    print("="*60 + "\n--- Starting Deforestation Model Training Workflow ---\n" + "="*60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    # Create output directory for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_folder = os.path.join(OUTPUT_FOLDER, run_timestamp)
    os.makedirs(run_output_folder, exist_ok=True)
    print(f"All outputs for this run will be saved in: {run_output_folder}")

    # Build dataset
    train_ds, val_ds, train_samples, val_samples = build_dataset(FILE_PATTERN, BATCH_SIZE, BUFFER_SIZE, VALIDATION_SPLIT)

    # Define models to train
    models_to_train = {
        "U-Net": build_unet_model(),
        # Add other models here for comparison, e.g.:
        # "ResU-Net": build_resunet_model(),
    }

    # Train models
    histories = {name: compile_and_train(model, train_ds, val_ds, EPOCHS, run_output_folder)
                 for name, model in models_to_train.items()}
    
    # Load best models for evaluation
    best_models = {name: tf.keras.models.load_model(os.path.join(run_output_folder, f'best_model_{name}.keras'))
                   for name in models_to_train.keys()}

    # Generate and save reports and visualizations
    accuracies = evaluate_and_report_metrics(best_models, val_ds, run_output_folder)
    save_visualizations(best_models, val_ds, run_output_folder)
    save_training_histories(histories, run_output_folder)

    # Log the experiment results
    end_time = time.time()
    total_run_time = f"{(end_time - start_time) / 60:.2f} minutes"
    
    best_model_name = max(accuracies, key=accuracies.get) if accuracies else "N/A"
    
    final_run_data = {
        'timestamp': run_timestamp,
        'models_trained': ", ".join(histories.keys()),
        'training_samples': train_samples,
        'validation_samples': val_samples,
        'batch_size': BATCH_SIZE,
        'epochs_run': EPOCHS,
        'best_model_name': best_model_name,
        'best_model_accuracy': accuracies.get(best_model_name, None),
        'notes': RUN_NOTES,
        'total_run_time': total_run_time
    }
    log_experiment(LOG_FILE, final_run_data)
    
    print("\n--- Workflow Complete ---")

if __name__ == '__main__':
    main()