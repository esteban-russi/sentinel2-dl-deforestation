"""
02.1-train-att-unet.py

Trains and evaluates an Attention U-Net for the 4-class land cover
segmentation task, with a final evaluation on a binary Forest vs.
Non-Forest task.

The model definition lives in `models.py`, while data loading,
training, evaluation, and experiment logging live in `pipeline.py`.
"""

from models import build_attention_unet_model, attention_gate
from pipeline import run_experiment


def main():
    """Trains the Attention U-Net."""
    run_experiment(build_attention_unet_model, show_summary=True)


if __name__ == '__main__':
    main()
