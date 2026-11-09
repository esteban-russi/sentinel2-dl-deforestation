"""
visualization.py

Visualization routines for model predictions, training histories,
and multi-temporal change detection maps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Configure Matplotlib for non-GUI backend
plt.switch_backend('Agg')

try:
    from .config import (
        NUM_CLASSES,
        CLASS_NAMES_4,
        VIS_PALETTE_4_CLASS,
        TCI_INDICES,
        CLASS_NAMES_CHANGE,
        CHANGE_MAP_PALETTE,
    )
except (ImportError, ValueError):
    from config import (
        NUM_CLASSES,
        CLASS_NAMES_4,
        VIS_PALETTE_4_CLASS,
        TCI_INDICES,
        CLASS_NAMES_CHANGE,
        CHANGE_MAP_PALETTE,
    )


def get_4_class_colormap():
    """Returns the ListedColormap and BoundaryNorm for 4-class Dynamic World visualizations."""
    colormap = ListedColormap(VIS_PALETTE_4_CLASS)
    bounds = np.arange(-0.5, NUM_CLASSES, 1)
    norm = BoundaryNorm(bounds, colormap.N)
    return colormap, norm


def save_visualizations(models_dict, val_data, output_folder, num_samples=5):
    """
    Saves model prediction visualizations to files using a custom, logical
    4-class color palette and legend.
    """
    print("\n--- Saving Final Prediction Visualizations ---")
    if not models_dict:
        return

    image_batch, label_batch = next(iter(val_data))
    true_labels = np.argmax(label_batch.numpy(), axis=-1)
    predictions = {
        name: np.argmax(model.predict(image_batch, verbose=0), axis=-1)
        for name, model in models_dict.items()
    }

    dw_colormap_4, dw_norm_4 = get_4_class_colormap()

    num_models = len(models_dict)
    fig_cols = 2 + num_models

    fig, axes = plt.subplots(num_samples, fig_cols, figsize=(5 * fig_cols, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(min(num_samples, image_batch.shape[0])):
        # Plot Input Image
        ax = axes[i, 0]
        tci_image = image_batch.numpy()[i][..., TCI_INDICES]
        ax.imshow(np.clip(tci_image, 0, 1) ** (1 / 1.8))
        ax.set_title(f"Input #{i+1}")
        ax.axis('off')

        # Plot Ground Truth
        ax = axes[i, 1]
        ax.imshow(true_labels[i], cmap=dw_colormap_4, norm=dw_norm_4)
        ax.set_title("Ground Truth")
        ax.axis('off')

        # Plot each model's prediction
        for j, (name, pred_labels) in enumerate(predictions.items()):
            ax = axes[i, 2 + j]
            ax.imshow(pred_labels[i], cmap=dw_colormap_4, norm=dw_norm_4)
            ax.set_title(f"{name} Prediction")
            ax.axis('off')

    patches = [Patch(color=VIS_PALETTE_4_CLASS[i], label=CLASS_NAMES_4[i]) for i in range(NUM_CLASSES)]
    fig.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title="4-Class Legend",
        fontsize='large',
        title_fontsize='x-large',
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    plt.savefig(os.path.join(output_folder, 'prediction_visualizations.png'), bbox_inches='tight')
    plt.close()


def save_training_histories(history_dict, output_folder):
    """Saves plots of training histories (Loss, Accuracy, IoU)."""
    print("\n--- Saving Training History Plots ---")
    plt.figure(figsize=(24, 6))
    plt.suptitle('Model Performance Comparison', fontsize=16)

    # Plot Loss
    plt.subplot(1, 3, 1)
    for name, history in history_dict.items():
        plt.plot(history.history['loss'], label=f'{name} Train Loss', linestyle='--')
        plt.plot(history.history['val_loss'], label=f'{name} Val Loss', linestyle='-')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    for name, history in history_dict.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train Accuracy', linestyle='--')
        plt.plot(history.history['val_accuracy'], label=f'{name} Val Accuracy', linestyle='-')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot IoU
    plt.subplot(1, 3, 3)
    for name, history in history_dict.items():
        iou_key = next((k for k in history.history if 'one_hot_io_u' in k and 'val_' not in k), None)
        val_iou_key = next((k for k in history.history if 'val_one_hot_io_u' in k), None)
        if iou_key and val_iou_key:
            plt.plot(history.history[iou_key], label=f'{name} Train IoU', linestyle='--')
            plt.plot(history.history[val_iou_key], label=f'{name} Val IoU', linestyle='-')
    plt.title('Training & Validation Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'training_histories.png'), bbox_inches='tight')
    plt.close()


def save_change_visualizations(change_map, raw_images_t1, raw_images_t2, output_path, num_samples_to_viz=10):
    """
    Generates and saves visual comparisons between T1, T2, and the resulting change map.
    """
    cmap_change = ListedColormap(CHANGE_MAP_PALETTE)
    patches = [
        Patch(color=color, label=name)
        for color, name in zip(CHANGE_MAP_PALETTE, CLASS_NAMES_CHANGE)
    ]

    fig, axes = plt.subplots(num_samples_to_viz, 3, figsize=(15, 5 * num_samples_to_viz))
    fig.suptitle('Sample Patches of Land Cover Change', fontsize=16, y=1.0)

    raw_t1_iter = raw_images_t1.as_numpy_iterator()
    raw_t2_iter = raw_images_t2.as_numpy_iterator()

    for i in range(num_samples_to_viz):
        try:
            image_t1_numpy = next(raw_t1_iter)
            image_t2_numpy = next(raw_t2_iter)

            tci_t1 = np.clip(image_t1_numpy[..., TCI_INDICES], 0, 1) ** (1 / 1.8)
            tci_t2 = np.clip(image_t2_numpy[..., TCI_INDICES], 0, 1) ** (1 / 1.8)

            change_patch = change_map[i]

            # Plot T1 Image
            axes[i, 0].imshow(tci_t1)
            axes[i, 0].set_title(f"Patch #{i} - T1 (Before)")
            axes[i, 0].axis('off')

            # Plot T2 Image
            axes[i, 1].imshow(tci_t2)
            axes[i, 1].set_title(f"Patch #{i} - T2 (After)")
            axes[i, 1].axis('off')

            # Plot Change Map
            axes[i, 2].imshow(change_patch, cmap=cmap_change, vmin=-1, vmax=1)
            axes[i, 2].set_title(f"Patch #{i} - Change Map")
            axes[i, 2].axis('off')

        except StopIteration:
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')

    fig.legend(handles=patches, bbox_to_anchor=(1.01, 0.9), loc='upper left', title="Change Types")
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualizations saved to {output_path}")
