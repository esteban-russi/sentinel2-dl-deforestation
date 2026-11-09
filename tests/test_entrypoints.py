import sys
from pathlib import Path
import importlib.util
from unittest.mock import MagicMock, patch

# Ensure repo root and src are in sys.path
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / 'src'
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(repo_root))

# Setup mocks
mock_tf = MagicMock()
mock_layers = MagicMock()
mock_tf.keras.layers = mock_layers
sys.modules.setdefault('tensorflow', mock_tf)
sys.modules.setdefault('tensorflow.keras', mock_tf.keras)
sys.modules.setdefault('tensorflow.keras.layers', mock_layers)


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_unet_entrypoint_main():
    mod = load_module_from_file('unet_train_mod', str(src_dir / '02.0-train-unet.py'))
    with patch.object(mod, 'run_experiment') as mock_run_exp:
        mod.main()
        assert mock_run_exp.called
        assert 'steps_per_epoch' in mock_run_exp.call_args[1]


def test_att_unet_entrypoint_main():
    mod = load_module_from_file('att_unet_train_mod', str(src_dir / '02.1-train-att-unet.py'))
    with patch.object(mod, 'run_experiment') as mock_run_exp:
        mod.main()
        assert mock_run_exp.called
        assert mock_run_exp.call_args[1].get('show_summary') is True


def test_change_detection_entrypoint_main():
    mod = load_module_from_file('change_detection_mod', str(src_dir / '03-change-detection-analysis.py'))
    with patch.object(mod, 'run_change_detection_analysis') as mock_run_cd:
        mod.main()
        assert mock_run_cd.called
