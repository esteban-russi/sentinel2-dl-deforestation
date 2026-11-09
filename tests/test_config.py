from pathlib import Path

import config


def test_config_constants():
    assert config.PATCH_SIZE == 256
    assert len(config.BANDS) == 9
    assert config.NUM_BANDS == 9
    assert config.TCI_BANDS == ['B4', 'B3', 'B2']
    assert config.TCI_INDICES == [config.BANDS.index(b) for b in config.TCI_BANDS]
    assert config.NUM_CLASSES == 4
    assert config.CLASS_NAMES_4 == ['Other', 'Trees', 'Grass', 'Shrub/Scrub']
    assert config.FOREST_CLASS_INDEX == 1
    assert config.BINARY_CLASS_NAMES == ['Non-Forest', 'Forest']
    assert config.CLASS_NAMES_CHANGE == ['Deforestation', 'No Change', 'Reforestation']
    assert len(config.CHANGE_MAP_PALETTE) == 3
    assert len(config.VIS_PALETTE_4_CLASS) == 4
    assert config.BATCH_SIZE == 5
    assert config.BUFFER_SIZE == 250
    assert config.EPOCHS == 50
    assert config.VALIDATION_SPLIT == 0.1
    assert config.SPLIT_SEED == 42
    assert config.VALIDATION_STEPS_PER_EPOCH == 15
    assert config.UNET_STEPS_PER_EPOCH == 410
    assert config.EXPORT_SCALE == 10
    assert config.PIXEL_RESOLUTION_M == 10


def test_paths_resolve_under_the_repository_root():
    repo_root = Path(__file__).resolve().parent.parent
    assert Path(config.REPO_ROOT) == repo_root
    assert Path(config.FILE_PATTERN) == repo_root / 'tfrecords' / 'colombia_production_data-*.tfrecord.gz'
    assert Path(config.LOG_FILE) == repo_root / 'run_outputs' / 'experiment_log.csv'
    assert Path(config.MODEL_FILE_PATH) == repo_root / 'best_models' / 'best_model_Attention_U-Net.keras'
    assert Path(config.INFERENCE_T1_FOLDER) == repo_root / 'inference_tfrecords_2021'
    assert Path(config.INFERENCE_T2_FOLDER) == repo_root / 'inference_tfrecords_2023'
