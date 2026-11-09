"""
training.py

Training routines, compilation, callbacks, and system hardware inspection.
"""

import os
import subprocess
import psutil
import tensorflow as tf

try:
    from .config import NUM_CLASSES, VALIDATION_STEPS_PER_EPOCH
except (ImportError, ValueError):
    from config import NUM_CLASSES, VALIDATION_STEPS_PER_EPOCH


def get_gpu_info():
    """Retrieves GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            text=True,
        )
        gpu_name, memory, driver = result.stdout.strip().split(',')
        return f"{gpu_name.strip()} ({memory.strip()}MiB; Driver: {driver.strip()})"
    except (FileNotFoundError, IndexError):
        return "N/A (nvidia-smi not found or failed)"


def get_cpu_info():
    """Retrieves CPU core information."""
    return f"{psutil.cpu_count(logical=False)} Physical Cores; {psutil.cpu_count(logical=True)} Total Cores"


def compile_and_train(
    model,
    train_data,
    val_data,
    epochs,
    output_folder,
    class_weights,
    steps_per_epoch=None,
    validation_steps=VALIDATION_STEPS_PER_EPOCH,
):
    """
    Compiles and trains the model, returning the history and the actual number of epochs run.
    """
    model_name = model.name
    print(f"\n--- Starting Workflow for Model: {model_name} ---")

    iou_metric = tf.keras.metrics.OneHotIoU(
        num_classes=NUM_CLASSES, target_class_ids=list(range(NUM_CLASSES))
    )

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', iou_metric],
    )

    print(f"Model '{model_name}' compiled successfully.")

    monitor_metric = f'val_{iou_metric.name}'

    # Reference to EarlyStopping callback to inspect stopped_epoch
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric, mode='max', patience=10, verbose=1, restore_best_weights=True
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_folder, f'best_model_{model_name}.keras'),
            save_best_only=True,
            monitor=monitor_metric,
            mode='max',
            verbose=1,
        ),
        early_stopping_callback,
    ]

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    # Capture the early stopping epoch
    stopped_epoch = early_stopping_callback.stopped_epoch
    if stopped_epoch > 0:
        epochs_actually_run = stopped_epoch + 1
        print(f"--- Early stopping triggered at epoch {epochs_actually_run} ---")
    else:
        epochs_actually_run = epochs
        print(f"--- Finished Training for {epochs} epochs (no early stopping) ---")

    return history, epochs_actually_run
