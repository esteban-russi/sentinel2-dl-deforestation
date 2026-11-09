"""
data.py

TFRecord parsing, class-weight calculation, and tf.data pipeline builders
for both model training (with remapped 4-class labels) and inference.
"""

import glob
import os
import numpy as np
import tensorflow as tf

from config import (
    PATCH_SIZE,
    BANDS,
    LABEL_BAND,
    NUM_CLASSES,
    CLASS_NAMES_4,
    BATCH_SIZE,
    BUFFER_SIZE,
    VALIDATION_SPLIT,
    SPLIT_SEED,
    FILE_PATTERN,
)


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


def calculate_class_weights(dataset, num_classes=NUM_CLASSES, class_names=CLASS_NAMES_4):
    """Calculates class weights based on inverse frequency from a dataset."""
    print("Calculating class weights from training data...")
    class_counts = np.zeros(num_classes)
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
    class_weights = 1 / (class_frequencies + 1e-6)  # Add epsilon to avoid division by zero
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * num_classes

    # Convert to a dictionary for model.fit()
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    print("Class Distribution and Weights:")
    for i in range(num_classes):
        print(f"  - {class_names[i]:<12}: Freq={class_frequencies[i]:.4f}, Weight={class_weight_dict[i]:.4f}")

    return class_weight_dict


def build_dataset(
    file_pattern=FILE_PATTERN,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    val_split=VALIDATION_SPLIT,
    split_seed=SPLIT_SEED,
):
    """Builds, splits, and batches the dataset for the 4-class task."""
    # Sort so the split does not depend on the order the filesystem lists files in.
    tfrecord_files = sorted(glob.glob(file_pattern))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found matching pattern: {file_pattern}")
    print(f"Found {len(tfrecord_files)} TFRecord files.")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP')
    dataset_size = raw_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

    # Shuffle ONCE with a fixed seed before splitting. take()/skip() on a dataset that
    # reshuffles on every iteration (the tf.data default) draws a different split each
    # epoch, so patches would move between training and validation and the validation
    # metrics would be computed partly on data the model was trained on.
    split_dataset = raw_dataset.shuffle(buffer_size, seed=split_seed, reshuffle_each_iteration=False)

    train_size = int((1 - val_split) * dataset_size)
    train_dataset = split_dataset.take(train_size).map(parse_tfrecord_4_class, num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset = split_dataset.skip(train_size).map(parse_tfrecord_4_class, num_parallel_calls=tf.data.AUTOTUNE)

    print(f"Splitting into {train_size} training samples and {dataset_size - train_size} validation samples.")

    # Calculate class weights ONLY on the training split
    class_weights = calculate_class_weights(train_dataset)

    # Reshuffle the training split every epoch; the validation split keeps a fixed order.
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Data pipeline built and split successfully.")

    return train_dataset, validation_dataset, train_size, dataset_size - train_size, class_weights


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
