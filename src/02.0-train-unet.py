"""
unet-train.py

Trains and evaluates the baseline U-Net for the 4-class land cover
segmentation task, with a final evaluation on a binary Forest vs.
Non-Forest task.

The data pipeline, training loop, evaluation, plots and experiment logging
are shared with the Attention U-Net and live in `pipeline.py`.
"""

# =============================================================================
# --- 0. Preamble and Imports ---
# =============================================================================
from tensorflow.keras import layers, Model

from pipeline import PATCH_SIZE, NUM_BANDS, NUM_CLASSES, BATCH_SIZE, run_experiment

# =============================================================================
# --- 1. Model Definition ---
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
# --- 2. Main Execution Block ---
# =============================================================================
def main():
    """Trains the baseline U-Net."""
    run_experiment(build_unet_model, steps_per_epoch=2051 // BATCH_SIZE)

if __name__ == '__main__':
    main()
