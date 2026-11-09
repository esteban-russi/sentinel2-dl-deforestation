import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure repo root and src are in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root))


def test_models_architecture_with_mocked_tf():
    # Setup mock tensorflow module
    mock_tf = MagicMock()
    mock_layers = MagicMock()
    mock_model_cls = MagicMock()

    # Configure mock layers to return callable tensor-like objects
    def fake_layer(*args, **kwargs):
        def apply(tensor):
            mock_out = MagicMock()
            mock_out.shape = (None, 256, 256, 32)
            return mock_out
        return apply

    mock_layers.Input = MagicMock(return_value=MagicMock(name='input_tensor'))
    mock_layers.Conv2D = MagicMock(side_effect=fake_layer)
    mock_layers.Dropout = MagicMock(side_effect=fake_layer)
    mock_layers.MaxPooling2D = MagicMock(side_effect=fake_layer)
    mock_layers.Conv2DTranspose = MagicMock(side_effect=fake_layer)
    mock_layers.concatenate = MagicMock(side_effect=lambda tensors: MagicMock(name='concat'))
    mock_layers.add = MagicMock(side_effect=lambda tensors: MagicMock(name='add'))
    mock_layers.multiply = MagicMock(side_effect=lambda tensors: MagicMock(name='multiply'))
    mock_layers.Activation = MagicMock(side_effect=fake_layer)

    mock_tf.keras.layers = mock_layers
    mock_tf.keras.Model = mock_model_cls

    sys.modules['tensorflow'] = mock_tf
    sys.modules['tensorflow.keras'] = mock_tf.keras
    sys.modules['tensorflow.keras.layers'] = mock_layers

    # Import models
    import importlib
    if 'models' in sys.modules:
        importlib.reload(sys.modules['models'])
    import models

    unet = models.build_unet_model(input_shape=(256, 256, 9), num_classes=4)
    assert mock_model_cls.called
    assert mock_model_cls.call_args[1]['name'] == 'U-Net'

    att_unet = models.build_attention_unet_model(input_shape=(256, 256, 9), num_classes=4)
    assert mock_model_cls.call_args[1]['name'] == 'Attention_U-Net'
