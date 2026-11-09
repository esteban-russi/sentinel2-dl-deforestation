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
    assert config.EXPORT_SCALE == 10
    assert config.PIXEL_RESOLUTION_M == 10
