"""
train-attention-unet.py

This script trains and evaluates an Attention U-Net model for the 4-class
land cover segmentation task, following the established project workflow.

Workflow:
1. Implements an Attention Gate module.
2. Defines the Attention U-Net architecture using this module.
3. Utilizes the existing 4-class data pipeline and class weighting.
4. Compiles, trains, and evaluates the Attention U-Net.
5. Performs post-processing for binary Forest vs. Non-Forest metrics.
6. Saves all results (model, plots, reports) to a new timestamped folder.
7. Logs a summary of the run to the central experiment log.
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import psutil

# Configure Matplotlib for non-GUI backend
plt.switch_backend('Agg')

# =============================================================================
# --- 1. Configuration & Constants ---
# =============================================================================


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

# --- 4-Class Label & Visualization Constants ---
NUM_CLASSES = 4
CLASS_NAMES_4 = ['Other', 'Trees', 'Grass', 'Shrub/Scrub']
BINARY_CLASS_NAMES = ['Non-Forest', 'Forest']

# --- Training Hyperparameters ---
BATCH_SIZE = 5
BUFFER_SIZE = 250
EPOCHS = 50
VALIDATION_SPLIT = 0.1
VALIDATION_STEPS_PER_EPOCH = 15


# --- Run-specific Notes ---
RUN_NOTES = f"full run; BUFFER_SIZE = {BUFFER_SIZE}; BATCH_SIZE = {BATCH_SIZE}; EPOCHS = {EPOCHS}; kept VAL_STEPS_PER_EPOCH = {VALIDATION_STEPS_PER_EPOCH}"
print("==================================================================")
print(RUN_NOTES)
print("==================================================================")


# =============================================================================
# --- 2. Data Pipeline (Identical to previous script) ---
# =============================================================================
# Functions: get_gpu_info, parse_tfrecord_4_class, calculate_class_weights, build_dataset
# These can be imported from a shared utils.py file in a larger project,
# but for a standalone script, it's fine to copy them here.

def get_gpu_info():
    """Retrieves GPU information using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, text=True)
        gpu_name, memory, driver = result.stdout.strip().split(',')
        return f"{gpu_name.strip()} ({memory.strip()}MiB; Driver: {driver.strip()})"
    except (FileNotFoundError, IndexError):
        return "N/A (nvidia-smi not found or failed)"

def parse_tfrecord_4_class(example_proto):
    """Parses a TFRecord, remaps 9 DW classes to our 4 target classes."""
    feature_description = {band: tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32) for band in BANDS}
    feature_description[LABEL_BAND] = tf.io.FixedLenFeature([], tf.string)
    example = tf.io.parse_single_example(example_proto, feature_description)
    features_list = [tf.reshape(example[band], [PATCH_SIZE, PATCH_SIZE]) for band in BANDS]
    image = tf.stack(features_list, axis=-1)
    label_decoded = tf.io.decode_raw(example[LABEL_BAND], tf.uint8)
    label_9_class = tf.reshape(label_decoded, [PATCH_SIZE, PATCH_SIZE])
    remapping = tf.constant([0, 1, 2, 0, 0, 3, 0, 0, 0], dtype=tf.int32)
    label_4_class = tf.gather(remapping, tf.cast(label_9_class, tf.int32))
    label_one_hot = tf.one_hot(label_4_class, depth=NUM_CLASSES)
    return image, label_one_hot

def calculate_class_weights(dataset):
    """Calculates class weights based on inverse frequency."""
    print("Calculating class weights from training data...")
    class_counts = np.zeros(NUM_CLASSES)
    for _, label_batch in dataset:
        labels_indices = tf.argmax(label_batch, axis=-1)
        unique, _, counts = tf.unique_with_counts(tf.reshape(labels_indices, [-1]))
        for idx, count in zip(unique.numpy(), counts.numpy()):
            class_counts[idx] += count
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    class_weights = 1 / (class_frequencies + 1e-6)
    class_weights = class_weights / np.sum(class_weights) * NUM_CLASSES
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("Class Distribution and Weights:")
    for i in range(NUM_CLASSES):
        print(f"  - {CLASS_NAMES_4[i]:<12}: Freq={class_frequencies[i]:.4f}, Weight={class_weight_dict[i]:.4f}")
        
    return class_weight_dict

def build_dataset(file_pattern, batch_size, buffer_size, val_split):
    """Builds, splits, and batches the dataset."""
    tfrecord_files = glob.glob(file_pattern)
    if not tfrecord_files: raise FileNotFoundError(f"No TFRecord files found: {file_pattern}")
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    dataset_size = raw_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    dataset = raw_dataset.map(parse_tfrecord_4_class, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size)
    train_size = int((1 - val_split) * dataset_size)
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    print(f"Splitting into {train_size} training samples and {dataset_size - train_size} validation samples.")
    class_weights = calculate_class_weights(train_dataset)
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Data pipeline built successfully.")
    return train_dataset, validation_dataset, train_size, dataset_size - train_size, class_weights

# =============================================================================
# --- 3. Model Definition: Attention U-Net ---
# =============================================================================

def attention_gate(g, x, inter_channel):
    """
    An Attention Gate module.
    
    Args:
        g: Gating signal from the deeper layer.
        x: Skip connection signal from the encoder.
        inter_channel: The number of channels for the intermediate convolution.
    """
    # Gating signal processing
    theta_g = layers.Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(g)
    
    # Skip connection signal processing
    phi_x = layers.Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(x)
    
    # Add signals
    add_gx = layers.add([theta_g, phi_x])
    add_gx = layers.Activation('relu')(add_gx)
    
    # Generate attention coefficients
    psi = layers.Conv2D(1, (1, 1), padding='same')(add_gx)
    psi = layers.Activation('sigmoid')(psi)
    
    # Apply attention coefficients to the skip connection signal
    return layers.multiply([x, psi])

def build_attention_unet_model(input_shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS), num_classes=NUM_CLASSES):
    """Builds an Attention U-Net model."""
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
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

    # --- Bottleneck ---
    b = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    b = layers.Dropout(0.3)(b)
    b = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(b)

    # --- Decoder with Attention Gates ---
    # Upsample to 64x64
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    # Apply attention gate to the skip connection c3
    a6 = attention_gate(g=u6, x=c3, inter_channel=64)
    u6 = layers.concatenate([u6, a6])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # Upsample to 128x128
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    # Apply attention gate to the skip connection c2
    a7 = attention_gate(g=u7, x=c2, inter_channel=32)
    u7 = layers.concatenate([u7, a7])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    # Upsample to 256x256
    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    # Apply attention gate to the skip connection c1
    a8 = attention_gate(g=u8, x=c1, inter_channel=16)
    u8 = layers.concatenate([u8, a8])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    # Output Layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c8)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="Attention_U-Net")
    return model

# =============================================================================
# --- 4. Training and Evaluation Functions (Identical to previous script) ---
# =============================================================================
# Functions: compile_and_train, evaluate_and_report_metrics, save_visualizations, save_training_histories, log_experiment
# These functions are generic and can be copied directly from your production script.
# For brevity, I will assume they are present and correct.

def compile_and_train(model, train_data, val_data, epochs, output_folder, class_weights):
    """
    Compiles and trains the model, returning the history and the actual number of epochs run.
    """
    model_name = model.name
    print(f"\n--- Starting Workflow for Model: {model_name} ---")
    
    iou_metric = tf.keras.metrics.OneHotIoU(num_classes=NUM_CLASSES, target_class_ids=list(range(NUM_CLASSES)))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', iou_metric])
    
    print(f"Model '{model_name}' compiled successfully.")
    
    monitor_metric = f'val_{iou_metric.name}'
    
    # --- We need a reference to the EarlyStopping callback to get its status later ---
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric, mode='max', patience=10, verbose=1, restore_best_weights=True
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(output_folder, f'best_model_{model_name}.keras'),
                                           save_best_only=True, monitor=monitor_metric, mode='max', verbose=1),
        early_stopping_callback # Add the callback instance here
    ]
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        # --- ADD THIS LINE ---
        validation_steps=VALIDATION_STEPS_PER_EPOCH, 
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )
    
    # --- NEW: Capture the early stopping epoch ---
    # The `stopped_epoch` attribute is 0 if it didn't stop early.
    stopped_epoch = early_stopping_callback.stopped_epoch
    if stopped_epoch > 0:
        # The epoch number is 0-indexed, so we add 1 for a human-readable count.
        epochs_actually_run = stopped_epoch + 1
        print(f"--- Early stopping triggered at epoch {epochs_actually_run} ---")
    else:
        # If it didn't stop early, it completed all configured epochs.
        epochs_actually_run = epochs
        print(f"--- Finished Training for {epochs} epochs (no early stopping) ---")

    # --- THE FIX: Return both the history object AND the calculated number of epochs ---
    return history, epochs_actually_run

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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=BINARY_CLASS_NAMES, yticklabels=BINARY_CLASS_NAMES)
        plt.title(f'Confusion Matrix - {name}', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_folder, f'binary_confusion_matrix_{name}.png'), bbox_inches='tight')
        plt.close()

    return binary_accuracies

    
def save_visualizations(models_dict, val_data, output_folder, num_samples=5):
    """
    Saves model prediction visualizations to files using a custom, logical
    4-class color palette and legend.
    """
    print("\n--- Saving Final Prediction Visualizations ---")
    if not models_dict: return

    image_batch, label_batch = next(iter(val_data))
    true_labels = np.argmax(label_batch.numpy(), axis=-1)
    predictions = {name: np.argmax(model.predict(image_batch, verbose=0), axis=-1) for name, model in models_dict.items()}
    
    # ------------------- MODIFICATION START -------------------
    # Define a logical color palette for the 4 classes
    # 0: Other -> Grey
    # 1: Trees -> Dark Green
    # 2: Grass -> Light Green
    # 3: Shrub/Scrub -> Olive/Yellow-Green
    VIS_PALETTE_4_CLASS = ['#9B9B9B', '#006400', '#88b053', '#dfc35a']
    
    # Create the custom colormap and normalization object
    dw_colormap_4 = ListedColormap(VIS_PALETTE_4_CLASS)
    bounds = np.arange(-0.5, NUM_CLASSES, 1) # NUM_CLASSES is 4
    dw_norm_4 = BoundaryNorm(bounds, dw_colormap_4.N)
    # -------------------- MODIFICATION END --------------------
    
    num_models = len(models_dict)
    fig_cols = 2 + num_models
    
    # Create the figure object
    fig, axes = plt.subplots(num_samples, fig_cols, figsize=(5 * fig_cols, 5 * num_samples))
    # Make sure axes is always a 2D array for consistent indexing
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(min(num_samples, image_batch.shape[0])):
        # Plot Input Image
        ax = axes[i, 0]
        tci_image = image_batch.numpy()[i][..., TCI_INDICES]
        ax.imshow(np.clip(tci_image, 0, 1) ** (1 / 1.8))
        ax.set_title(f"Input #{i+1}"); ax.axis('off')

        # Plot Ground Truth
        ax = axes[i, 1]
        # Apply the new custom colormap and normalization
        ax.imshow(true_labels[i], cmap=dw_colormap_4, norm=dw_norm_4)
        ax.set_title("Ground Truth"); ax.axis('off')
        
        # Plot each model's prediction
        for j, (name, pred_labels) in enumerate(predictions.items()):
            ax = axes[i, 2 + j]
            # Apply the new custom colormap and normalization
            ax.imshow(pred_labels[i], cmap=dw_colormap_4, norm=dw_norm_4)
            ax.set_title(f"{name} Prediction"); ax.axis('off')
            
    # Create a single, shared legend for the entire figure
    patches = [Patch(color=VIS_PALETTE_4_CLASS[i], label=CLASS_NAMES_4[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="4-Class Legend", fontsize='large', title_fontsize='x-large')

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1.0]) # The right boundary is reduced
    plt.savefig(os.path.join(output_folder, 'prediction_visualizations.png'), bbox_inches='tight')
    plt.close()

def save_training_histories(history_dict, output_folder):
    """Saves plots of training histories, now including accuracy."""
    print("\n--- Saving Training History Plots ---")
    # --- NEW: Number of plots is now 3 ---
    plt.figure(figsize=(24, 6))
    plt.suptitle('Model Performance Comparison', fontsize=16)

    # Plot Loss
    plt.subplot(1, 3, 1)
    for name, history in history_dict.items():
        plt.plot(history.history['loss'], label=f'{name} Train Loss', linestyle='--')
        plt.plot(history.history['val_loss'], label=f'{name} Val Loss', linestyle='-')
    plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    for name, history in history_dict.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train Accuracy', linestyle='--')
        plt.plot(history.history['val_accuracy'], label=f'{name} Val Accuracy', linestyle='-')
    plt.title('Training & Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    # Plot IoU
    plt.subplot(1, 3, 3)
    for name, history in history_dict.items():
        iou_key = next((k for k in history.history if 'one_hot_io_u' in k and 'val_' not in k), None)
        val_iou_key = next((k for k in history.history if 'val_one_hot_io_u' in k), None)
        if iou_key and val_iou_key:
            plt.plot(history.history[iou_key], label=f'{name} Train IoU', linestyle='--')
            plt.plot(history.history[val_iou_key], label=f'{name} Val IoU', linestyle='-')
    plt.title('Training & Validation Mean IoU'); plt.xlabel('Epoch'); plt.ylabel('Mean IoU')
    plt.legend(); plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    """Main function to orchestrate the Attention U-Net workflow."""
    print("="*60 + "\n--- Starting Attention U-Net Model Workflow ---\n" + "="*60)
    print("Run notes:", RUN_NOTES)
    # --- Setup ---
    gpu_info_str = get_gpu_info()
    cpu_info_str = f"{psutil.cpu_count(logical=False)} Cores, {psutil.cpu_count(logical=True)} Threads"
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_folder = os.path.join(OUTPUT_FOLDER, run_timestamp)
    os.makedirs(run_output_folder, exist_ok=True)
    print(f"Hardware: {gpu_info_str} | {cpu_info_str}")
    print(f"Saving outputs to: {run_output_folder}")

    # --- Data ---
    train_ds, val_ds, train_samples, val_samples, class_weights = build_dataset(
        FILE_PATTERN, BATCH_SIZE, BUFFER_SIZE, VALIDATION_SPLIT
    )

    # --- Model Training ---
    # Here we only define and train the Attention U-Net
    models_to_train = {
        "Attention_U-Net": build_attention_unet_model(num_classes=NUM_CLASSES),
    }
    
    start_time = time.time()
    histories = {}
    epochs_run_dict = {}

    for name, model in models_to_train.items():
        model.summary()
        history, epochs_run = compile_and_train(model, train_ds, val_ds, EPOCHS, run_output_folder, class_weights)
        histories[name] = history
        epochs_run_dict[name] = epochs_run
    
    # --- Evaluation & Reporting ---
    best_models = {
        name: tf.keras.models.load_model(os.path.join(run_output_folder, f'best_model_{name}.keras'))
        for name in models_to_train.keys()
    }
    best_model_name = "Attention_U-Net"
    
    binary_accuracies = evaluate_and_report_metrics(best_models, val_ds, run_output_folder)
    save_visualizations(best_models, val_ds, run_output_folder)
    save_training_histories(histories, run_output_folder)
    
    # --- Final Logging ---
    end_time = time.time()
    total_run_time = f"{(end_time - start_time) / 60:.2f}"
     
    final_run_data = {
        'timestamp': run_timestamp,
        'models_trained': ", ".join(histories.keys()),
        'training_samples': train_samples,
        'validation_samples': val_samples,
        'batch_size': BATCH_SIZE,
        'epochs_run': str(EPOCHS)+"; stopped_at: "+ str(epochs_run_dict.get(best_model_name, None)),
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