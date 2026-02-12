import subprocess
import sys

try:
    from skimage.graph import MCP_Geometric  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image', '--quiet'])

from .plugin import RoeDeerCorridorDemoPlugin


def classFactory(iface):
    return RoeDeerCorridorDemoPlugin(iface)
