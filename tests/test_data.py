import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')

import data  # noqa: E402  (imported after the TensorFlow skip check)

PATCH_SIZE = 4  # Tiny patches keep the synthetic TFRecords fast to write and parse


@pytest.fixture(autouse=True)
def small_patches(monkeypatch):
    monkeypatch.setattr(data, 'PATCH_SIZE', PATCH_SIZE)


def make_example(patch_id, label_9_class):
    """Builds one serialized patch. Every band holds the patch id so it can be traced after parsing."""
    features = {
        band: tf.train.Feature(float_list=tf.train.FloatList(value=[float(patch_id)] * PATCH_SIZE**2))
        for band in data.BANDS
    }
    features[data.LABEL_BAND] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[np.asarray(label_9_class, dtype=np.uint8).tobytes()])
    )
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def write_tfrecords(folder, num_patches, num_files=2):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    for file_index in range(num_files):
        path = folder / f'colombia_production_data-{file_index}.tfrecord.gz'
        with tf.io.TFRecordWriter(str(path), options) as writer:
            for patch_id in range(file_index, num_patches, num_files):
                label = np.arange(PATCH_SIZE**2) % 9  # every Dynamic World class appears
                writer.write(make_example(patch_id, label))
    return str(folder / 'colombia_production_data-*.tfrecord.gz')


def patch_ids(dataset):
    """Returns the patch ids in a batched (image, label) dataset, in iteration order."""
    return [int(image[0, 0, 0]) for image_batch, _ in dataset for image in image_batch.numpy()]


def test_parse_remaps_dynamic_world_classes_to_four_classes():
    label_9_class = np.arange(PATCH_SIZE**2) % 9
    image, label_one_hot = data.parse_tfrecord_4_class(make_example(7, label_9_class))

    assert image.shape == (PATCH_SIZE, PATCH_SIZE, len(data.BANDS))
    assert np.all(image.numpy() == 7)
    expected = np.array([0, 1, 2, 0, 0, 3, 0, 0, 0])[label_9_class].reshape(PATCH_SIZE, PATCH_SIZE)
    np.testing.assert_array_equal(np.argmax(label_one_hot.numpy(), axis=-1), expected)


def test_train_and_validation_splits_never_share_patches(tmp_path):
    num_patches = 40
    pattern = write_tfrecords(tmp_path, num_patches)

    train_ds, val_ds, train_size, val_size, _ = data.build_dataset(
        pattern, batch_size=4, buffer_size=num_patches, val_split=0.25, split_seed=42
    )

    assert (train_size, val_size) == (30, 10)
    val_epochs = [patch_ids(val_ds) for _ in range(3)]
    train_epochs = [patch_ids(train_ds) for _ in range(3)]

    # The validation split is identical, in the same order, on every epoch.
    assert val_epochs[0] == val_epochs[1] == val_epochs[2]
    # The training split holds the same patches every epoch, never a validation patch.
    for train_ids in train_epochs:
        assert len(train_ids) == train_size
        assert set(train_ids) == set(train_epochs[0])
        assert set(train_ids).isdisjoint(val_epochs[0])
    assert set(train_epochs[0]) | set(val_epochs[0]) == set(range(num_patches))


def test_split_is_reproducible_for_a_fixed_seed(tmp_path):
    pattern = write_tfrecords(tmp_path, num_patches=40)

    def validation_ids():
        _, val_ds, *_ = data.build_dataset(pattern, batch_size=4, buffer_size=40, val_split=0.25, split_seed=42)
        return patch_ids(val_ds)

    assert validation_ids() == validation_ids()


def test_inference_files_are_read_in_name_order(tmp_path, monkeypatch):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    file_names = [f'inference_data_2021-{i:05d}.tfrecord.gz' for i in range(3)]
    for patch_id, file_name in enumerate(file_names):
        with tf.io.TFRecordWriter(str(tmp_path / file_name), options) as writer:
            writer.write(make_example(patch_id, np.zeros(PATCH_SIZE**2)))
    # Simulate a filesystem that lists the files in a different order.
    listed_in_reverse = [str(tmp_path / name) for name in reversed(file_names)]
    monkeypatch.setattr(data.glob, 'glob', lambda pattern: listed_in_reverse)

    _, raw_images = data.build_inference_dataset(str(tmp_path), batch_size=2)

    assert [int(image[0, 0, 0]) for image in raw_images.as_numpy_iterator()] == [0, 1, 2]
