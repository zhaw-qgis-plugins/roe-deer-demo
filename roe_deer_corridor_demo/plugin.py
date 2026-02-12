from qgis.core import QgsApplication

from .provider import RoeDeerCorridorDemoProvider


class RoeDeerCorridorDemoPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = RoeDeerCorridorDemoProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
