def test_pipeline_exports_and_compatibility(mock_tensorflow):
    import pipeline

    expected_symbols = [
        'DATA_FOLDER',
        'OUTPUT_FOLDER',
        'FILE_PATTERN',
        'LOG_FILE',
        'PATCH_SIZE',
        'BANDS',
        'LABEL_BAND',
        'TCI_BANDS',
        'NUM_BANDS',
        'TCI_INDICES',
        'NUM_CLASSES',
        'CLASS_NAMES_4',
        'BINARY_CLASS_NAMES',
        'BATCH_SIZE',
        'BUFFER_SIZE',
        'EPOCHS',
        'VALIDATION_SPLIT',
        'SPLIT_SEED',
        'VALIDATION_STEPS_PER_EPOCH',
        'RUN_NOTES',
        'get_gpu_info',
        'parse_tfrecord_4_class',
        'calculate_class_weights',
        'build_dataset',
        'compile_and_train',
        'evaluate_and_report_metrics',
        'save_visualizations',
        'save_training_histories',
        'log_experiment',
        'run_experiment',
    ]

    for symbol in expected_symbols:
        assert hasattr(pipeline, symbol), f"pipeline is missing symbol: {symbol}"
