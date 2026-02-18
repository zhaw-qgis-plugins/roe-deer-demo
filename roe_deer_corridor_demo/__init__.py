from .plugin import RoeDeerCorridorDemoPlugin


def classFactory(iface):
    return RoeDeerCorridorDemoPlugin(iface)
