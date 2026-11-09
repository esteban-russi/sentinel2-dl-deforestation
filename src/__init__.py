"""
sentinel2-dl-deforestation

End-to-end deep learning package for forest change detection in Colombia
using Sentinel-2 satellite imagery and Google Earth Engine / Dynamic World.
"""

from . import config

try:
    from . import data
except ImportError:
    data = None

try:
    from . import models
except ImportError:
    models = None

try:
    from . import training
except ImportError:
    training = None

try:
    from . import evaluation
except ImportError:
    evaluation = None

try:
    from . import visualization
except ImportError:
    visualization = None

try:
    from . import change_detection
except ImportError:
    change_detection = None

try:
    from . import gee_export
except ImportError:
    gee_export = None

try:
    from . import pipeline
except ImportError:
    pipeline = None

__all__ = [
    'config',
    'data',
    'models',
    'training',
    'evaluation',
    'visualization',
    'change_detection',
    'gee_export',
    'pipeline',
]
