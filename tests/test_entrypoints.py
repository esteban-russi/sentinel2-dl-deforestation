import importlib.util
from unittest.mock import patch

import pytest

from conftest import SRC_DIR


def load_script(file_name):
    """Imports an entry script whose file name is not a valid module name."""
    spec = importlib.util.spec_from_file_location(file_name.replace('-', '_').replace('.', '_'), SRC_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _mock_tf(mock_tensorflow):
    return mock_tensorflow


def test_unet_entrypoint_main():
    mod = load_script('02.0-train-unet.py')
    with patch.object(mod, 'run_experiment') as mock_run_exp:
        mod.main()
        assert mock_run_exp.called
        assert mock_run_exp.call_args[1]['steps_per_epoch'] == 410


def test_att_unet_entrypoint_main():
    mod = load_script('02.1-train-att-unet.py')
    with patch.object(mod, 'run_experiment') as mock_run_exp:
        mod.main()
        assert mock_run_exp.called
        assert mock_run_exp.call_args[1].get('show_summary') is True


def test_change_detection_entrypoint_main():
    mod = load_script('03-change-detection-analysis.py')
    with patch.object(mod, 'run_change_detection_analysis') as mock_run_cd:
        mod.main()
        assert mock_run_cd.called
