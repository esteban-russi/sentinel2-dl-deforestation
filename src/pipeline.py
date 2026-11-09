"""
pipeline.py

Shared data pipeline, training, evaluation and reporting facade for the
land cover segmentation models. Re-exports all pipeline components and
provides the `run_experiment` orchestrator.
"""

import os
import time
from datetime import datetime
import pandas as pd
import tensorflow as tf

from config import (
    DATA_FOLDER,
    OUTPUT_FOLDER,
    FILE_PATTERN,
    LOG_FILE,
    PATCH_SIZE,
    BANDS,
    LABEL_BAND,
    TCI_BANDS,
    NUM_BANDS,
    TCI_INDICES,
    NUM_CLASSES,
    CLASS_NAMES_4,
    BINARY_CLASS_NAMES,
    BATCH_SIZE,
    BUFFER_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    SPLIT_SEED,
    VALIDATION_STEPS_PER_EPOCH,
    RUN_NOTES,
)
from data import (
    parse_tfrecord_4_class,
    calculate_class_weights,
    build_dataset,
)
from training import (
    get_gpu_info,
    get_cpu_info,
    compile_and_train,
)
from evaluation import evaluate_and_report_metrics
from visualization import (
    save_visualizations,
    save_training_histories,
)

# pipeline.py used to define everything below itself. Keep exposing the same
# names so existing imports of the form `from pipeline import X` keep working.
__all__ = [
    'DATA_FOLDER',
    'OUTPUT_FOLDER',
    'FILE_PATTERN',
    'LOG_FILE',
    'PATCH_SIZE',
    'BANDS',
    'LABEL_BAND',
    'TCI_BANDS',
    'NUM_BANDS',
    'TCI_INDICES',
    'NUM_CLASSES',
    'CLASS_NAMES_4',
    'BINARY_CLASS_NAMES',
    'BATCH_SIZE',
    'BUFFER_SIZE',
    'EPOCHS',
    'VALIDATION_SPLIT',
    'SPLIT_SEED',
    'VALIDATION_STEPS_PER_EPOCH',
    'RUN_NOTES',
    'get_gpu_info',
    'get_cpu_info',
    'parse_tfrecord_4_class',
    'calculate_class_weights',
    'build_dataset',
    'compile_and_train',
    'evaluate_and_report_metrics',
    'save_visualizations',
    'save_training_histories',
    'log_experiment',
    'run_experiment',
]


def log_experiment(log_file_path, run_data):
    """Appends a summary of the run to a CSV log file."""
    new_log_df = pd.DataFrame([run_data])
    write_header = not os.path.exists(log_file_path)
    new_log_df.to_csv(log_file_path, mode='a', header=write_header, index=False)
    print("\n" + "="*50 + f"\nSUCCESS: Run results logged to '{log_file_path}'\n" + "="*50)


def run_experiment(build_model_fn, steps_per_epoch=None, show_summary=False):
    """
    Runs the full workflow for one model: builds the dataset, trains the model
    returned by `build_model_fn`, evaluates the best checkpoint, saves plots and
    appends a summary row to the experiment log.
    """
    print("="*60 + "\n--- Starting Deforestation Model Workflow ---\n" + "="*60)
    print("Run notes:", RUN_NOTES)
    # --- Get and Log Hardware Info ---
    gpu_info_str = get_gpu_info()
    cpu_info_str = get_cpu_info()
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

    # --- Model Training ---
    model = build_model_fn(num_classes=NUM_CLASSES)
    model_name = model.name
    if show_summary:
        model.summary()

    start_time = time.time()
    history, epochs_run = compile_and_train(
        model,
        train_ds,
        val_ds,
        EPOCHS,
        run_output_folder,
        class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_steps=VALIDATION_STEPS_PER_EPOCH,
    )
    histories = {model_name: history}

    # --- Evaluation & Reporting ---
    best_models = {
        model_name: tf.keras.models.load_model(os.path.join(run_output_folder, f'best_model_{model_name}.keras'))
    }

    binary_accuracies = evaluate_and_report_metrics(best_models, val_ds, run_output_folder)
    save_visualizations(best_models, val_ds, run_output_folder)
    save_training_histories(histories, run_output_folder)

    # --- Final Logging ---
    end_time = time.time()
    total_run_time = f"{(end_time - start_time) / 60:.2f}"

    final_run_data = {
        'timestamp': run_timestamp,
        'models_trained': model_name,
        'training_samples': train_samples,
        'validation_samples': val_samples,
        'batch_size': BATCH_SIZE,
        'epochs_run': str(EPOCHS) + "; stopped_at: " + str(epochs_run),
        'best_model_name': model_name,
        'best_model_val_accuracy': binary_accuracies.get(model_name, None),
        'notes': RUN_NOTES,
        'total_run_time': total_run_time,
        'cpu_info': cpu_info_str,
        'gpu_info': gpu_info_str,
    }
    log_experiment(LOG_FILE, final_run_data)

    print("\n--- Workflow Complete ---")
