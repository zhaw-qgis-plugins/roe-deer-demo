from qgis.core import QgsProcessingProvider

from .roe_deer_demo import RoeDeerDemo


class RoeDeerCorridorDemoProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(RoeDeerDemo())

    def id(self):
        return "roedeer"

    def name(self):
        return "Roe Deer Corridor Demo"

    def longName(self):
        return "Roe Deer Corridor Demo"
