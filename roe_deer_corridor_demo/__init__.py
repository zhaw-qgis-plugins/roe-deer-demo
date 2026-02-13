try:
    from skimage.graph import MCP_Geometric  # noqa: F401
except ImportError:
    raise ImportError(
        "This plugin requires scikit-image. "
        "See the plugin documentation for installation instructions."
    )

from .plugin import RoeDeerCorridorDemoPlugin


def classFactory(iface):
    return RoeDeerCorridorDemoPlugin(iface)
