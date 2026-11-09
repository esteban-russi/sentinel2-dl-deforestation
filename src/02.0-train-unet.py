"""
02.0-train-unet.py

Trains and evaluates the baseline U-Net for the 4-class land cover
segmentation task, with a final evaluation on a binary Forest vs.
Non-Forest task.

The model definition lives in `models.py`, while data loading,
training, evaluation, and experiment logging live in `pipeline.py`.
"""

try:
    from .config import BATCH_SIZE
    from .models import build_unet_model
    from .pipeline import run_experiment
except (ImportError, ValueError):
    from config import BATCH_SIZE
    from models import build_unet_model
    from pipeline import run_experiment


def main():
    """Trains the baseline U-Net."""
    run_experiment(build_unet_model, steps_per_epoch=2051 // BATCH_SIZE)


if __name__ == '__main__':
    main()
