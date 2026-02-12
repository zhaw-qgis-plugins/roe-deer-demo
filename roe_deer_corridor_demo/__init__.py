import os
import subprocess
import sys

try:
    from skimage.graph import MCP_Geometric  # noqa: F401
except ImportError:
    # Install into the QGIS profile python dir (writable, already on sys.path)
    _target = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        'scikit-image', '--target', _target, '--quiet',
    ])
    from skimage.graph import MCP_Geometric  # noqa: F401

from .plugin import RoeDeerCorridorDemoPlugin


def classFactory(iface):
    return RoeDeerCorridorDemoPlugin(iface)
