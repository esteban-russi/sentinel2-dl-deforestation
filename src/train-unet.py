"""
train_models_4class.py

This script performs an end-to-end training and evaluation workflow for a
4-class land cover segmentation task, with a final evaluation on a binary
Forest vs. Non-Forest task.

Workflow:
1. Configures constants for the 4-class problem.
2. Implements a tf.data pipeline to remap 9 classes to 4.
3. Calculates class weights to handle imbalance in the training set.
4. Defines, compiles, and trains a U-Net model on the 4-class task.
5. Evaluates the model, performing post-processing to get binary metrics.
6. Saves all results (models, plots, reports) to a timestamped folder.
7. Logs a summary, including hardware info, to a CSV file.
"""

# =============================================================================
# --- 0. Preamble and Imports ---
# =============================================================================
import os
import time
import glob
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import psutil

# Configure Matplotlib for non-GUI backend
plt.switch_backend('Agg')

# =============================================================================
# --- 1. Configuration & Constants (Refactored for 4-Class Task) ---
# =============================================================================
# --- Run-specific Notes ---
RUN_NOTES = "U-Net 4-class training (Trees, Grass, Shrub, Other) with class weights. Binary Forest/Non-Forest final eval."

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

# --- NEW: 4-Class Label & Visualization Constants ---
# Original Dynamic World classes: 0:water, 1:trees, 2:grass, 3:flooded_veg, 4:crops, 5:shrub, 6:built, 7:bare, 8:snow
# Our new mapping:
# 0: Other (0, 3, 4, 6, 7, 8)
# 1: Trees (1)
# 2: Grass (2)
# 3: Shrub/Scrub (5)
NUM_CLASSES = 4
CLASS_NAMES_4 = ['Other', 'Trees', 'Grass', 'Shrub/Scrub']
# Binary mapping for final evaluation
BINARY_CLASS_NAMES = ['Non-Forest', 'Forest']

# --- Training Hyperparameters ---
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EPOCHS = 50
VALIDATION_SPLIT = 0.1

# =============================================================================
# --- 2. Data Pipeline (Refactored for 4-Class Task) ---
# =============================================================================
def get_gpu_info():
    """Retrieves GPU information using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, text=True)
        gpu_name, memory, driver = result.stdout.strip().split(',')
        return f"{gpu_name.strip()} ({memory.strip()}MiB, Driver: {driver.strip()})"
    except (FileNotFoundError, IndexError):
        return "N/A (nvidia-smi not found or failed)"

def parse_tfrecord_4_class(example_proto):
    """
    Parses a TFRecord, remaps 9 DW classes to our 4 target classes,
    and returns the image and one-hot label.
    """
    feature_description = {band: tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32) for band in BANDS}
    feature_description[LABEL_BAND] = tf.io.FixedLenFeature([], tf.string)
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    features_list = [tf.reshape(example[band], [PATCH_SIZE, PATCH_SIZE]) for band in BANDS]
    image = tf.stack(features_list, axis=-1)
    
    label_decoded = tf.io.decode_raw(example[LABEL_BAND], tf.uint8)
    label_9_class = tf.reshape(label_decoded, [PATCH_SIZE, PATCH_SIZE])
    label_9_class = tf.cast(label_9_class, tf.int32)
    
    # --- Remapping Logic ---
    # Create a mapping tensor. Index corresponds to old class, value to new class.
    # Old: 0,1,2,3,4,5,6,7,8 -> New: 0,1,2,0,0,3,0,0,0
    remapping = tf.constant([0, 1, 2, 0, 0, 3, 0, 0, 0], dtype=tf.int32)
    label_4_class = tf.gather(remapping, label_9_class)
    
    label_one_hot = tf.one_hot(label_4_class, depth=NUM_CLASSES)
    
    return image, label_one_hot

def calculate_class_weights(dataset):
    """Calculates class weights based on inverse frequency from a dataset."""
    print("Calculating class weights from training data...")
    class_counts = np.zeros(NUM_CLASSES)
    # Iterate through the dataset to count pixels for each class
    for _, label_batch in dataset:
        # labels are one-hot encoded, so argmax gives the class index
        labels_indices = tf.argmax(label_batch, axis=-1)
        unique, _, counts = tf.unique_with_counts(tf.reshape(labels_indices, [-1]))
        for idx, count in zip(unique.numpy(), counts.numpy()):
            class_counts[idx] += count

    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    
    # Inverse frequency weighting
    class_weights = 1 / (class_frequencies + 1e-6) # Add epsilon to avoid division by zero
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * NUM_CLASSES

    # Convert to a dictionary for model.fit()
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("Class Distribution and Weights:")
    for i in range(NUM_CLASSES):
        print(f"  - {CLASS_NAMES_4[i]:<12}: Freq={class_frequencies[i]:.4f}, Weight={class_weight_dict[i]:.4f}")
        
    return class_weight_dict

def build_dataset(file_pattern, batch_size, buffer_size, val_split):
    """Builds, splits, and batches the dataset for the 4-class task."""
    tfrecord_files = glob.glob(file_pattern)
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found matching pattern: {file_pattern}")
    print(f"Found {len(tfrecord_files)} TFRecord files.")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    dataset_size = raw_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    
    # Map with the new 4-class parsing function
    dataset = raw_dataset.map(parse_tfrecord_4_class, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size)
    
    train_size = int((1 - val_split) * dataset_size)
    #train_size = 10
    
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    
    print(f"Splitting into {train_size} training samples and {dataset_size - train_size} validation samples.")
    
    # Calculate class weights ONLY on the training split
    class_weights = calculate_class_weights(train_dataset)
    
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("Data pipeline built and split successfully.")
    
    return train_dataset, validation_dataset, train_size, dataset_size - train_size, class_weights

# =============================================================================
# --- 3. Model Definition ---
# =============================================================================
def build_unet_model(input_shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS), num_classes=NUM_CLASSES):
    """Builds a U-Net model, now outputting num_classes=4."""
    inputs = layers.Input(shape=input_shape)
    # ... [U-Net architecture is identical, only the final layer changes] ...
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
    # The final layer now outputs 4 channels for our 4 classes
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c8)
    model = Model(inputs=[inputs], outputs=[outputs], name="U-Net")
    return model

# =============================================================================
# --- 4. Training and Evaluation Functions (Refactored for 4-Class & Binary) ---
# =============================================================================
def compile_and_train(model, train_data, val_data, epochs, output_folder, class_weights):
    """Compiles and trains the model, using class weights and monitoring val_IoU."""
    model_name = model.name
    print(f"\n--- Starting Workflow for Model: {model_name} ---")
    
    # Find the name of the IoU metric for monitoring
    iou_metric = tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=list(range(NUM_CLASSES)))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', iou_metric])
    
    print(f"Model '{model_name}' compiled successfully.")
    
    # Monitor validation IoU for saving the best model and for early stopping
    monitor_metric = f'val_{iou_metric.name}'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_folder, f'best_model_{model_name}.keras'),
                                           save_best_only=True, monitor=monitor_metric, mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, mode='max', patience=10, verbose=1, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
        class_weight=class_weights, # Pass the calculated class weights
        verbose=2
    )
    print(f"--- Finished Training: {model_name} ---")
    return history

def evaluate_and_report_metrics(models_dict, validation_dataset, output_folder):
    """
    Evaluates on the 4-class task and then post-processes for a final
    binary (Forest vs. Non-Forest) evaluation.
    """
    print("\n" + "="*50 + "\n--- GENERATING FINAL EVALUATION METRICS ---\n" + "="*50)
    
    all_true_labels_4_class, all_pred_labels_dict = [], {name: [] for name in models_dict.keys()}
    
    for image_batch, label_batch in validation_dataset:
        true_labels = np.argmax(label_batch.numpy(), axis=-1).flatten()
        all_true_labels_4_class.extend(true_labels)
        for name, model in models_dict.items():
            pred_batch = model.predict(image_batch, verbose=0)
            pred_labels = np.argmax(pred_batch, axis=-1).flatten()
            all_pred_labels_dict[name].extend(pred_labels)
            
    # --- Post-processing for Binary Classification ---
    # Remap both true and predicted labels to Forest (1) vs. Non-Forest (0)
    # In our 4-class setup: 'Trees' is class 1, all others are non-forest.
    all_true_labels_binary = [1 if label == 1 else 0 for label in all_true_labels_4_class]
    
    binary_accuracies = {}
    for name, pred_labels_4_class in all_pred_labels_dict.items():
        print("\n" + "-"*20 + f" REPORT FOR MODEL: {name} " + "-"*20)
        
        pred_labels_binary = [1 if label == 1 else 0 for label in pred_labels_4_class]
        
        report = classification_report(all_true_labels_binary, pred_labels_binary, target_names=BINARY_CLASS_NAMES)
        print("\nBinary Classification Report (Forest vs. Non-Forest):")
        print(report)
        
        accuracy = accuracy_score(all_true_labels_binary, pred_labels_binary)
        print(f"Overall Binary Pixel Accuracy: {accuracy:.4f}")
        binary_accuracies[name] = accuracy
        
        cm = confusion_matrix(all_true_labels_binary, pred_labels_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=BINARY_CLASS_NAMES, yticklabels=BINARY_CLASS_NAMES)
        plt.title(f'Confusion Matrix - {name}', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_folder, f'binary_confusion_matrix_{name}.png'), bbox_inches='tight')
        plt.close()

    return binary_accuracies

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
    """Main function to orchestrate the workflow."""
    start_time = time.time()
    
    print("="*60 + "\n--- Starting Deforestation Model Workflow ---\n" + "="*60)
    
    # --- Get and Log Hardware Info ---
    gpu_info_str = get_gpu_info()
    cpu_info_str = f"{psutil.cpu_count(logical=False)} Physical Cores, {psutil.cpu_count(logical=True)} Total Cores"
    print(f"GPU Info: {gpu_info_str}")
    print(f"CPU Info: {cpu_info_str}")

    # Create output directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_folder = os.path.join(OUTPUT_FOLDER, run_timestamp)
    os.makedirs(run_output_folder, exist_ok=True)
    print(f"All outputs will be saved in: {run_output_folder}")

    # Build dataset and get class weights
    train_ds, val_ds, train_samples, val_samples, class_weights = build_dataset(
        FILE_PATTERN, BATCH_SIZE, BUFFER_SIZE, VALIDATION_SPLIT
    )

    # Define models to train (make sure num_classes is passed correctly)
    models_to_train = {
        "U-Net": build_unet_model(num_classes=NUM_CLASSES),
    }

    # Train models
    histories = {
        name: compile_and_train(model, train_ds, val_ds, EPOCHS, run_output_folder, class_weights)
        for name, model in models_to_train.items()
    }
    
    # Load best models for evaluation
    best_models = {
        name: tf.keras.models.load_model(os.path.join(run_output_folder, f'best_model_{name}.keras'))
        for name in models_to_train.keys()
    }

    # Evaluate models on the final binary task
    binary_accuracies = evaluate_and_report_metrics(best_models, val_ds, run_output_folder)
    save_visualizations(best_models, val_ds, run_output_folder) # Still useful for debugging
    save_training_histories(histories, run_output_folder)

    # Log the experiment results
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    total_run_time = f"{int(minutes)}:{seconds:.0f}"
    print(f"Total Run Time: {total_run_time}")

    
    best_model_name = max(binary_accuracies, key=binary_accuracies.get) if binary_accuracies else "N/A"
    
    final_run_data = {
        'timestamp': run_timestamp,
        'models_trained': ", ".join(histories.keys()),
        'training_samples': train_samples,
        'validation_samples': val_samples,
        'batch_size': BATCH_SIZE,
        'epochs_run': EPOCHS,
        'best_model_name': best_model_name,
        'best_model_val_accuracy': binary_accuracies.get(best_model_name, None),
        'notes': RUN_NOTES,
        'total_run_time': total_run_time,
        'cpu_info': cpu_info_str,
        'gpu_info': gpu_info_str
    }
    log_experiment(LOG_FILE, final_run_data)
    
    print("\n--- Workflow Complete ---")

if __name__ == '__main__':
    main()