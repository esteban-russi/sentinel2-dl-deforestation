"""
models.py

Modular neural network components and architectures for land cover semantic segmentation:
- conv_block: Standard double Conv2D + Dropout block.
- attention_gate: Additive soft attention gate module.
- build_unet_model: Standard U-Net architecture.
- build_attention_unet_model: Attention U-Net architecture.
"""

from tensorflow.keras import layers, Model

from config import PATCH_SIZE, NUM_BANDS, NUM_CLASSES


def conv_block(
    inputs,
    filters,
    dropout_rate=0.1,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal',
    padding='same',
):
    """
    Standard convolutional block consisting of:
    Conv2D -> Dropout -> Conv2D
    """
    c = layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(inputs)
    if dropout_rate > 0:
        c = layers.Dropout(dropout_rate)(c)
    c = layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(c)
    return c


def attention_gate(g, x, inter_channel):
    """
    An Attention Gate module.

    Args:
        g: Gating signal from the deeper layer.
        x: Skip connection signal from the encoder.
        inter_channel: The number of channels for the intermediate convolution.

    Returns:
        Tensor of the same shape as x with attention gating applied.
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


def build_unet_model(
    input_shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS),
    num_classes=NUM_CLASSES,
    name="U-Net",
):
    """Builds a standard U-Net model outputting num_classes channels."""
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
    c1 = conv_block(inputs, 32, dropout_rate=0.1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64, dropout_rate=0.1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128, dropout_rate=0.2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- Bottleneck ---
    b = conv_block(p3, 256, dropout_rate=0.3)

    # --- Decoder ---
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u6 = layers.concatenate([u6, c3])
    c6 = conv_block(u6, 128, dropout_rate=0.2)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c2])
    c7 = conv_block(u7, 64, dropout_rate=0.1)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c1])
    c8 = conv_block(u8, 32, dropout_rate=0.1)

    # The final layer outputs num_classes channels with softmax activation
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c8)
    model = Model(inputs=[inputs], outputs=[outputs], name=name)
    return model


def build_attention_unet_model(
    input_shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS),
    num_classes=NUM_CLASSES,
    name="Attention_U-Net",
):
    """Builds an Attention U-Net model outputting num_classes channels."""
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
    c1 = conv_block(inputs, 32, dropout_rate=0.1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64, dropout_rate=0.1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128, dropout_rate=0.2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- Bottleneck ---
    b = conv_block(p3, 256, dropout_rate=0.3)

    # --- Decoder with Attention Gates ---
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    a6 = attention_gate(g=u6, x=c3, inter_channel=64)
    u6 = layers.concatenate([u6, a6])
    c6 = conv_block(u6, 128, dropout_rate=0.2)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    a7 = attention_gate(g=u7, x=c2, inter_channel=32)
    u7 = layers.concatenate([u7, a7])
    c7 = conv_block(u7, 64, dropout_rate=0.1)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    a8 = attention_gate(g=u8, x=c1, inter_channel=16)
    u8 = layers.concatenate([u8, a8])
    c8 = conv_block(u8, 32, dropout_rate=0.1)

    # Output Layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c8)
    model = Model(inputs=[inputs], outputs=[outputs], name=name)
    return model
