"""
Name: Roe Deer Corridor Demo
Group: Wildlife Modelling
Description: Calculates LCPs and shows ALL endpoints (Reachable vs Unreachable).
"""

import processing
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
                       QgsProcessingException,
                       QgsVectorLayer,
                       QgsApplication,
                       QgsProcessingUtils)
import numpy as np
from osgeo import gdal

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

        ds = gdal.Open(raster_path)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype('float32')
        nodata = band.GetNoDataValue()
        gt = ds.GetGeoTransform()
        x_origin = gt[0]
        px_w = gt[1]
        y_origin = gt[3]
        px_h = gt[5]
        c_start = int((start_pt.x() - x_origin) / px_w)
        r_start = int((start_pt.y() - y_origin) / px_h)
        ds = None  # close dataset

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

        # 7. GRASS availability check
        r_cost_id = (
            "grass7:r.cost"
            if QgsApplication.processingRegistry().algorithmById("grass7:r.cost")
            else "grass:r.cost"
            if QgsApplication.processingRegistry().algorithmById("grass:r.cost")
            else None
        )
        r_path_id = r_cost_id.replace("r.cost", "r.path") if r_cost_id else None
        if not r_cost_id:
            raise QgsProcessingException(
                "GRASS GIS is required but not available in this QGIS installation."
            )

        # Step A: r.cost
        feedback.pushInfo("Running r.cost (cumulative cost surface)...")
        cost_tmp = QgsProcessingUtils.generateTempFilename("rcost.tif")
        dir_tmp  = QgsProcessingUtils.generateTempFilename("rdir.tif")

        processing.run(r_cost_id, {
            'input':                           raster_layer,
            'start_coordinates':               f"{start_pt.x()},{start_pt.y()}",
            'output':                          cost_tmp,
            'outdir':                          dir_tmp,
            'GRASS_REGION_PARAMETER':          raster_layer,
            'GRASS_REGION_CELLSIZE_PARAMETER': 0,
        }, context=context, feedback=feedback)

        # Step B: Read cost raster
        cost_ds   = gdal.Open(cost_tmp)
        cost_band = cost_ds.GetRasterBand(1)
        cost_data = cost_band.ReadAsArray().astype('float64')
        cost_nd   = cost_band.GetNoDataValue()
        cost_ds   = None

        # Step C: Classify targets
        reachable     = []  # (id, x, y, cost)
        count_unreach = 0

        for i in range(len(t_rows)):
            if feedback.isCanceled(): break
            r, c = int(t_rows[i]), int(t_cols[i])
            x    = x_origin + (c * px_w) + (px_w / 2)
            y    = y_origin + (r * px_h) + (px_h / 2)
            cost = float(cost_data[r, c])
            is_bad = (np.isinf(cost) or np.isnan(cost)
                      or (cost_nd is not None and abs(cost - cost_nd) < 1e-6))
            if is_bad:
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                feat.setAttributes([i, "Unreachable", 999999.9])
                sink_pts.addFeature(feat)
                count_unreach += 1
            else:
                reachable.append((i, x, y, cost))

        feedback.pushInfo(f"Reachable: {len(reachable)}, Unreachable: {count_unreach}")

        if not reachable:
            return {self.OUTPUT_LINES: id_lines, self.OUTPUT_POINTS: id_pts}

        # Step D: Build memory layer of reachable targets
        mem_layer = QgsVectorLayer(f"Point?crs={dest_crs.authid()}", "targets", "memory")
        prov = mem_layer.dataProvider()
        prov.addAttributes([QgsField("target_id", QVariant.Int), QgsField("cost", QVariant.Double)])
        mem_layer.updateFields()
        feats = []
        for (tid, x, y, cost) in reachable:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            f.setAttributes([tid, cost])
            feats.append(f)
        prov.addFeatures(feats)

        # Step E: r.path
        feedback.pushInfo("Running r.path (least-cost paths)...")
        paths_tmp      = QgsProcessingUtils.generateTempFilename("paths.gpkg")
        rpath_ras_tmp  = QgsProcessingUtils.generateTempFilename("rpath.tif")

        processing.run(r_path_id, {
            'input':                           dir_tmp,
            'format':                          0,    # 0 = degree (r.cost default)
            'start_points':                    mem_layer,
            'vector_path':                     paths_tmp,
            'raster_path':                     rpath_ras_tmp,
            'GRASS_REGION_PARAMETER':          raster_layer,
            'GRASS_REGION_CELLSIZE_PARAMETER': 0,
        }, context=context, feedback=feedback)

        # Step F: Read paths, match to targets, emit features
        paths_layer = QgsVectorLayer(paths_tmp, "paths", "ogr")
        # Build lookup: target (x,y) → (id, cost)
        cost_lookup = {(tx, ty): (tid, tc) for (tid, tx, ty, tc) in reachable}
        half_px = abs(px_w) / 2

        count_reach = 0
        for path_feat in paths_layer.getFeatures():
            if feedback.isCanceled(): break
            geom = path_feat.geometry()
            if geom.isNull() or geom.isEmpty(): continue

            # r.path traces target → source; first vertex = the target point
            verts = list(geom.vertices())
            if not verts: continue
            v = verts[0]

            # Snap to nearest known reachable target within half-pixel tolerance
            best_key = min(cost_lookup, key=lambda k: (k[0] - v.x())**2 + (k[1] - v.y())**2)
            if abs(best_key[0] - v.x()) > half_px or abs(best_key[1] - v.y()) > half_px:
                continue  # no match — skip orphan path
            tid, tc = cost_lookup[best_key]

            f_line = QgsFeature()
            f_line.setGeometry(geom)
            f_line.setAttributes([tid, tc])
            sink_lines.addFeature(f_line)

            f_pt = QgsFeature()
            f_pt.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*best_key)))
            f_pt.setAttributes([tid, "Reachable", tc])
            sink_pts.addFeature(f_pt)
            count_reach += 1

        feedback.pushInfo(f"Finished. Reachable: {count_reach}, Blocked: {count_unreach}")
        return {self.OUTPUT_LINES: id_lines, self.OUTPUT_POINTS: id_pts}
