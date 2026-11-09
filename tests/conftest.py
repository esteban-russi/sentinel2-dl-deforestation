import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
PROJECT_MODULES = [path.stem for path in SRC_DIR.glob('*.py')]


def _is_project_module(module):
    module_file = getattr(module, '__file__', None)
    return module_file is not None and Path(module_file).resolve().parent == SRC_DIR


@pytest.fixture
def mock_tensorflow(monkeypatch):
    """Replace tensorflow with a MagicMock for the duration of one test.

    Project modules imported while the mock is active hold references to it,
    so they are dropped from sys.modules afterwards and any module that was
    already imported is restored. Other tests therefore never see the mock.
    """
    tf = MagicMock(name='tensorflow')
    monkeypatch.setitem(sys.modules, 'tensorflow', tf)
    monkeypatch.setitem(sys.modules, 'tensorflow.keras', tf.keras)
    monkeypatch.setitem(sys.modules, 'tensorflow.keras.layers', tf.keras.layers)
    for name in PROJECT_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    modules_before = set(sys.modules)

    yield tf

    for name in set(sys.modules) - modules_before:
        if _is_project_module(sys.modules[name]):
            del sys.modules[name]
