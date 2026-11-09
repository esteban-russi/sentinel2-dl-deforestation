"""
evaluation.py

Evaluation, classification metrics generation (4-class and binary Forest vs Non-Forest),
and confusion matrix plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import BINARY_CLASS_NAMES


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

        report = classification_report(
            all_true_labels_binary, pred_labels_binary, target_names=BINARY_CLASS_NAMES
        )
        print("\nBinary Classification Report (Forest vs. Non-Forest):")
        print(report)

        accuracy = accuracy_score(all_true_labels_binary, pred_labels_binary)
        print(f"Overall Binary Pixel Accuracy: {accuracy:.4f}")
        binary_accuracies[name] = accuracy

        cm = confusion_matrix(all_true_labels_binary, pred_labels_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=BINARY_CLASS_NAMES,
            yticklabels=BINARY_CLASS_NAMES,
        )
        plt.title(f'Confusion Matrix - {name}', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_folder, f'binary_confusion_matrix_{name}.png'), bbox_inches='tight')
        plt.close()

    return binary_accuracies
