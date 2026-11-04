"""
train-attention-unet.py

Trains and evaluates an Attention U-Net for the 4-class land cover
segmentation task, with a final evaluation on a binary Forest vs.
Non-Forest task.

The data pipeline, training loop, evaluation, plots and experiment logging
are shared with the baseline U-Net and live in `pipeline.py`.
"""

# =============================================================================
# --- 0. Preamble and Imports ---
# =============================================================================
from tensorflow.keras import layers, Model

from pipeline import PATCH_SIZE, NUM_BANDS, NUM_CLASSES, run_experiment

# =============================================================================
# --- 1. Model Definition: Attention U-Net ---
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
# --- 2. Main Execution Block ---
# =============================================================================
def main():
    """Trains the Attention U-Net."""
    run_experiment(build_attention_unet_model, show_summary=True)

if __name__ == '__main__':
    main()
