###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################


from __future__ import print_function
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators.opBlockedArrayCache import OpBlockedArrayCache
from lazyflow.operators.classifierOperators import OpPixelwiseClassifierPredict
from lazyflow.operators import OpMultiArraySlicer2, OpMaxChannelIndicatorOperator

from ilastik.utility.operatorSubView import OperatorSubView


class OpDLClassification(Operator):
    """
    Top-level operator for pixel classification
    """

    name = "OpDLClassification"
    category = "Top-level"

    # Graph inputs
    Classifier = InputSlot()
    InputImage = InputSlot()
    NumClasses = InputSlot()
    BlockShape = InputSlot()

    FreezePredictions = InputSlot(stype="bool", value=True, nonlane=True)

    PredictionProbabilities = OutputSlot()
    CachedPredictionProbabilities = OutputSlot()
    PredictionProbabilityChannels = OutputSlot(level=1)
    SegmentationChannels = OutputSlot(level=1)

    # Gui only (not part of the pipeline)
    ModelPath = InputSlot()  # Path
    FullModel = InputSlot(value=[])  # When full model serialization is enabled
    Batch_Size = InputSlot(value=1)
    Block_Size = InputSlot(value=16 * 1024)
    Window_Size = InputSlot(value=256)  # the neural net will break up large images (of size up to Block_size) into smaller overlapping windows of size Window_Size
    SaveFullModel = InputSlot(stype="bool", value=False, nonlane=True)

    def __init__(self, *args, **kwargs):

        super(OpDLClassification, self).__init__(*args, **kwargs)

        self.predict = OpPixelwiseClassifierPredict(parent=self)
        self.predict.name = "OpClassifierPredict"
        self.predict.Image.connect(self.InputImage)
        self.predict.Classifier.connect(self.Classifier)
        self.predict.LabelsCount.connect(self.NumClasses)
        self.PredictionProbabilities.connect(self.predict.PMaps)

        self.prediction_cache = OpBlockedArrayCache(parent=self)
        self.prediction_cache.name = "BlockedArrayCache"
        self.prediction_cache.inputs["Input"].connect(self.predict.PMaps)
        self.prediction_cache.BlockShape.connect(self.BlockShape)
        self.prediction_cache.inputs["fixAtCurrent"].connect(self.FreezePredictions)
        self.CachedPredictionProbabilities.connect(self.prediction_cache.Output)

        self.opPredictionSlicer = OpMultiArraySlicer2(parent=self)
        self.opPredictionSlicer.name = "opPredictionSlicer"
        self.opPredictionSlicer.Input.connect(self.prediction_cache.Output)
        self.opPredictionSlicer.AxisFlag.setValue("c")
        self.PredictionProbabilityChannels.connect(self.opPredictionSlicer.Slices)

        self.opSegmentor = OpMaxChannelIndicatorOperator(parent=self)
        self.opSegmentor.Input.connect(self.prediction_cache.Output)

        self.opSegmentationSlicer = OpMultiArraySlicer2(parent=self)
        self.opSegmentationSlicer.name = "opSegmentationSlicer"
        self.opSegmentationSlicer.Input.connect(self.opSegmentor.Output)
        self.opSegmentationSlicer.AxisFlag.setValue("c")
        self.SegmentationChannels.connect(self.opSegmentationSlicer.Slices)

    def propagateDirty(self, slot, subindex, roi):
        """
        PredicitionProbabilityChannels is called when the visibility of SetupLayers is changed
        """
        self.PredictionProbabilityChannels.setDirty(slice(None))

    def addLane(self, laneIndex):
        pass

    def removeLane(self, laneIndex, finalLength):
        pass

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)
