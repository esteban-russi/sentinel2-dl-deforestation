from unittest.mock import MagicMock


def fake_layer(*args, **kwargs):
    """Stands in for a Keras layer: calling it on a tensor returns a new mock tensor."""
    def apply(tensor):
        output = MagicMock()
        output.shape = (None, 256, 256, 32)
        return output
    return apply


def test_models_architecture_with_mocked_tf(mock_tensorflow):
    layers = mock_tensorflow.keras.layers
    layers.Input = MagicMock(return_value=MagicMock(name='input_tensor'))
    for layer_name in ['Conv2D', 'Dropout', 'MaxPooling2D', 'Conv2DTranspose', 'Activation']:
        setattr(layers, layer_name, MagicMock(side_effect=fake_layer))
    for merge_name in ['concatenate', 'add', 'multiply']:
        setattr(layers, merge_name, MagicMock(side_effect=lambda tensors, n=merge_name: MagicMock(name=n)))
    model_cls = mock_tensorflow.keras.Model

    import models

    models.build_unet_model(input_shape=(256, 256, 9), num_classes=4)
    assert model_cls.called
    assert model_cls.call_args[1]['name'] == 'U-Net'

    models.build_attention_unet_model(input_shape=(256, 256, 9), num_classes=4)
    assert model_cls.call_args[1]['name'] == 'Attention_U-Net'
