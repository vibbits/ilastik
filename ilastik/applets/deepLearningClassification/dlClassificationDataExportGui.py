###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################


from lazyflow.operators.generic import OpMultiArraySlicer2
from volumina.api import createDataSource, AlphaModulatedLayer
from ilastik.applets.dataExport.dataExportGui import DataExportGui, DataExportLayerViewerGui


class DLClassificationDataExportGui(DataExportGui):
    """
    A subclass of the generic data export gui that creates custom layer viewers.
    """

    def createLayerViewer(self, opLane):
        return DLClassificationResultsViewer(self.parentApplet, opLane)


class DLClassificationResultsViewer(DataExportLayerViewerGui):
    """
    SubClass for the DataExport viewerGui to show the layers correctly
    """

    def __init__(self, *args, **kwargs):
        super(DLClassificationResultsViewer, self).__init__(*args, **kwargs)

    def setupLayers(self):
        layers = []
        opLane = self.topLevelOperatorView

        # This code depends on a specific order for the export slots.
        # If those change, update this function!
        selection_names = opLane.SelectionNames.value

        # see comment above
        for name, expected in zip(selection_names[0:1], ["Probabilities"]):
            assert name.startswith(expected), "The Selection Names don't match the expected selection names."

        selection = selection_names[opLane.InputSelection.value]

        if selection.startswith("Probabilities"):
            exportedLayers = self._initPredictionLayers(opLane.ImageToExport)
            for layer in exportedLayers:
                layer.visible = True
                layer.name = layer.name + "- Exported"
            layers += exportedLayers

        # If available, also show the raw data layer
        rawSlot = opLane.FormattedRawData
        if rawSlot.ready():
            rawLayer = self.createStandardLayerFromSlot(rawSlot)
            rawLayer.name = "Raw Data"
            rawLayer.visible = True
            rawLayer.opacity = 1.0
            layers.append(rawLayer)

        return layers

    def _initPredictionLayers(self, predictionSlot):
        opLane = self.topLevelOperatorView
        layers = []

        # Use a slicer to provide a separate slot for each channel layer
        opSlicer = OpMultiArraySlicer2(parent=opLane.viewed_operator().parent)
        opSlicer.Input.connect(predictionSlot)
        opSlicer.AxisFlag.setValue("c")

        for channel, predictionSlot in enumerate(opSlicer.Slices):
            if predictionSlot.ready():
                predictsrc = createDataSource(predictionSlot)
                predictLayer = AlphaModulatedLayer(predictsrc, range=(0.0, 1.0), normalize=(0.0, 1.0))
                predictLayer.opacity = 0.25
                predictLayer.visible = True

                def setPredLayerName(n, predictLayer_=predictLayer, initializing=False):
                    """
                    function for setting the names for every Channel
                    """
                    if not initializing and predictLayer_ not in self.layerstack:
                        # This layer has been removed from the layerstack already.
                        # Don't touch it.
                        return
                    newName = "Prediction for %s" % n
                    predictLayer_.name = newName

                setPredLayerName(channel, initializing=True)

                layers.append(predictLayer)

        return layers
