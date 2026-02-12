"""
Name: Roe Deer Corridor Demo
Group: Wildlife Modelling
Description: Calculates LCPs and shows ALL endpoints (Reachable vs Unreachable).
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterPoint,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSink,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsField,
                       QgsFields,
                       QgsCoordinateTransform,
                       QgsWkbTypes,
                       QgsProcessingException)
import numpy as np
import rasterio
from skimage.graph import MCP_Geometric

class RoeDeerDemo(QgsProcessingAlgorithm):
    INPUT_RASTER = 'INPUT_RASTER'
    START_POINT = 'START_POINT'
    GRID_SPACING = 'GRID_SPACING'
    OUTPUT_LINES = 'OUTPUT_LINES'
    OUTPUT_POINTS = 'OUTPUT_POINTS'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return RoeDeerDemo()

    def name(self):
        return 'roedeer_demo'

    def displayName(self):
        return self.tr('Roe Deer Corridor Demo')

    def group(self):
        return self.tr('Wildlife Modelling')

    def groupId(self):
        return 'wildlife_modelling'

    def shortHelpString(self):
        return self.tr("Generates LCPs. Marks unreachable endpoints to help Demo connectivity issues.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_RASTER, self.tr('Resistance Surface')))
        self.addParameter(QgsProcessingParameterPoint(self.START_POINT, self.tr('Start Point')))
        self.addParameter(QgsProcessingParameterNumber(self.GRID_SPACING, self.tr('Grid Spacing (m)'), defaultValue=2000))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_LINES, self.tr('Corridor Paths')))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_POINTS, self.tr('All Endpoints (Status)')))

    def processAlgorithm(self, parameters, context, feedback):
        # 1. Inputs
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        source_point = self.parameterAsPoint(parameters, self.START_POINT, context)
        grid_spacing = self.parameterAsInt(parameters, self.GRID_SPACING, context)

        if raster_layer is None: raise QgsProcessingException("Invalid Raster")

        # 2. Transform Click to Raster CRS
        source_crs = context.project().crs()
        dest_crs = raster_layer.crs()
        transform = QgsCoordinateTransform(source_crs, dest_crs, context.project())
        start_pt = transform.transform(source_point)

        # 3. Read Data
        raster_path = raster_layer.source()
        feedback.pushInfo(f"Reading: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            # Load as float32 to handle Infinite costs
            data = src.read(1).astype('float32')
            r_transform = src.transform
            nodata = src.nodata
            
            # Get Start Index
            r_start, c_start = src.index(start_pt.x(), start_pt.y())
            
            # Geometry Helpers
            x_origin = r_transform[2]
            y_origin = r_transform[5]
            px_w = r_transform[0]
            px_h = r_transform[4]

        # 4. Validate Start Point
        if not (0 <= r_start < data.shape[0] and 0 <= c_start < data.shape[1]):
             raise QgsProcessingException("Click outside raster extent.")

        start_val = data[r_start, c_start]
        feedback.pushInfo(f"Start Pixel Resistance: {start_val}")
        
        if start_val <= 0 or (nodata is not None and start_val == nodata) or np.isnan(start_val):
             raise QgsProcessingException(f"Invalid Start! Resistance is {start_val}. Must be > 0.")

        # 5. Setup Outputs
        # Lines
        f_lines = QgsFields()
        f_lines.append(QgsField("id", QVariant.Int))
        f_lines.append(QgsField("cost", QVariant.Double))
        (sink_lines, id_lines) = self.parameterAsSink(parameters, self.OUTPUT_LINES, context, f_lines, QgsWkbTypes.LineString, dest_crs)

        # Points (With Status Field)
        f_pts = QgsFields()
        f_pts.append(QgsField("id", QVariant.Int))
        f_pts.append(QgsField("status", QVariant.String)) # 'Start', 'Reachable', 'Unreachable'
        f_pts.append(QgsField("cost", QVariant.Double))
        (sink_pts, id_pts) = self.parameterAsSink(parameters, self.OUTPUT_POINTS, context, f_pts, QgsWkbTypes.Point, dest_crs)

        # Add Start Point Marker
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(start_pt))
        feat.setAttributes([-1, "Start", 0.0])
        sink_pts.addFeature(feat)

        # 6. Find Targets (Grid)
        feedback.pushInfo("Scanning for targets...")
        pixel_res = px_w
        step = int(grid_spacing / pixel_res)
        if step < 1: step = 1
        
        rows = np.arange(0, data.shape[0], step)
        cols = np.arange(0, data.shape[1], step)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        
        mask = (data[rr, cc] == 1.0)
        t_rows = rr[mask]
        t_cols = cc[mask]
        
        feedback.pushInfo(f"Found {len(t_rows)} potential targets.")

        # 7. Pathfinding
        feedback.pushInfo("Running MCP...")
        mcp = MCP_Geometric(data, fully_connected=True)
        costs, _ = mcp.find_costs(starts=[(r_start, c_start)])

        # 8. Process Results
        count_reach = 0
        count_unreach = 0

        for i in range(len(t_rows)):
            if feedback.isCanceled(): break
            
            r, c = t_rows[i], t_cols[i]
            cost = float(costs[r, c])
            
            # Calculate Coordinate for this target
            x = x_origin + (c * px_w) + (px_w/2)
            y = y_origin + (r * px_h) + (px_h/2)
            pt_geom = QgsPointXY(x, y)

            # CHECK 1: Is it reachable?
            if np.isinf(cost) or np.isnan(cost):
                # CASE: UNREACHABLE
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(pt_geom))
                feat.setAttributes([i, "Unreachable", 999999.9])
                sink_pts.addFeature(feat)
                count_unreach += 1
                continue

            # CHECK 2: Can we trace it?
            path = mcp.traceback((r, c))
            if not path or len(path) < 2:
                # CASE: Too close or error
                continue

            # CASE: REACHABLE
            # Draw Line
            line_pts = []
            for pr, pc in path[::-1]:
                lx = x_origin + (pc * px_w) + (px_w/2)
                ly = y_origin + (pr * px_h) + (px_h/2)
                line_pts.append(QgsPointXY(lx, ly))
            
            f_line = QgsFeature()
            f_line.setGeometry(QgsGeometry.fromPolylineXY(line_pts))
            f_line.setAttributes([i, cost])
            sink_lines.addFeature(f_line)
            
            # Draw Endpoint
            f_pt = QgsFeature()
            f_pt.setGeometry(QgsGeometry.fromPointXY(pt_geom))
            f_pt.setAttributes([i, "Reachable", cost])
            sink_pts.addFeature(f_pt)
            count_reach += 1

        feedback.pushInfo(f"Finished. Reachable: {count_reach}, Blocked: {count_unreach}")
        return {self.OUTPUT_LINES: id_lines, self.OUTPUT_POINTS: id_pts}