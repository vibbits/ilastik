###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################

from __future__ import print_function

import logging

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators.opBlockedArrayCache import OpBlockedArrayCache
from lazyflow.operators.classifierOperators import OpPixelwiseClassifierPredict
from lazyflow.operators import OpMultiArraySlicer2, OpMaxChannelIndicatorOperator

from ilastik.utility.operatorSubView import OperatorSubView

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    PredictionProbabilities = OutputSlot()                # Result straight from the neural network classifier

    # For the Classification GUI layers
    PredictionProbabilityChannels = OutputSlot(level=1)   # Used for the probability layers in the classification gui
    SegmentationChannels = OutputSlot(level=1)            # Used for the segmentation layers in the classification gui

    # For the Export GUI layers.
    # During export these outputs will be passed through an OpMultiArraySlicer2 to split it into channels.
    # IMPROVEME:
    # It would make much more sense to use PredictionProbabilityChannels and SegmentationChannels
    # instead, which are split in exactly the same way. However, we ran into issues with level 1 versus level 0 slot
    # mismatches and a missing wrapper. This is probably easy to solve, if only we understood the export code's software design better...
    CachedPredictionProbabilities = OutputSlot()          # Used for the probability layer in the export gui. It is the cached version of PredictionProbabilities. We must use it during Export instead of PredictionProbabilities to avoid that the export code asks for predictions with small block sizes - remember we want to feed the full image to the neural network. Now the Export code will query the cache for small blocks, but that is fine since the cache has already calculated that result earlier on using the full image.
    Segmentation = OutputSlot()                           # Used for the segmentation layer in the export gui. Segmentation is an n-channel binary image.

    # Gui only (not part of the pipeline)
    ModelPath = InputSlot()  # Path (a single string) to the neural network saved via pytorch
    # FullModel = InputSlot(value=[])  # When full model serialization is enabled
    # SaveFullModel = InputSlot(stype="bool", value=False, nonlane=True)
    Batch_Size = InputSlot(value=1)
    Block_Size = InputSlot(value=16 * 1024)
    Window_Size = InputSlot(value=256)  # the neural net will break up large images (of size up to Block_size) into smaller overlapping windows of size Window_Size

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

        self.opSegmentor = OpMaxChannelIndicatorOperator(parent=self)  # IMPROVEME: this "max channel" works for binary classification, but not when we have more than 2 classes
        self.opSegmentor.Input.connect(self.prediction_cache.Output)
        self.Segmentation.connect(self.opSegmentor.Output)

        self.opSegmentationSlicer = OpMultiArraySlicer2(parent=self)
        self.opSegmentationSlicer.name = "opSegmentationSlicer"
        self.opSegmentationSlicer.Input.connect(self.opSegmentor.Output)
        self.opSegmentationSlicer.AxisFlag.setValue("c")
        self.SegmentationChannels.connect(self.opSegmentationSlicer.Slices)

    def propagateDirty(self, slot, subindex, roi):
        """
        PredicitionProbabilityChannels is called when the visibility of SetupLayers is changed
        """
        logger.debug(f"OpDLClassification.propagateDirty slot={slot} subindex={subindex} roi={roi} -> setting PredictionProbabilityChannels dirty")
        self.PredictionProbabilityChannels.setDirty(slice(None))
        # FIXME/CHECKME Why is this needed? To force calculation of the complete output?
        # FIXME/CHECKME Why only for PredictionProbabilityChannels? Why not for SegmentationChannels, for example?

    def addLane(self, laneIndex):
        pass

    def removeLane(self, laneIndex, finalLength):
        pass

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)
