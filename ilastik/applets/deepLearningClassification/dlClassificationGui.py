###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################

import os
import logging
from functools import partial
from collections import OrderedDict

import numpy
import sys

from neuralnets.util.tools import load_net

from volumina import colortables
from volumina.api import createDataSource, AlphaModulatedLayer
from volumina.utility import preferences

from PyQt5 import uic
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QStackedWidget,
    QMessageBox,
    QFileDialog,
    QMenu,
    QLineEdit,
    QDialogButtonBox,
    QVBoxLayout,
    QDialog,
    QCheckBox,
    QLabel,
)


from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.widgets.labelListView import Label
from ilastik.widgets.labelListModel import LabelListModel
from ilastik.config import cfg as ilastik_config

from lazyflow.classifiers import DeepLearningLazyflowClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ParameterDlg(QDialog):
    """
    simple window for setting parameters
    """

    def __init__(self, topLevelOperator, parent):
        super(QDialog, self).__init__(parent=parent)

        self.topLevelOperator = topLevelOperator

        buttonbox = QDialogButtonBox(Qt.Horizontal, parent=self)
        buttonbox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonbox.accepted.connect(self.add_Parameters)
        buttonbox.rejected.connect(self.reject)

        self.block_edit = QLineEdit(self)
        self.block_edit.setText(str(topLevelOperator.Block_Size.value))
        self.batch_edit = QLineEdit(self)
        self.batch_edit.setText(str(topLevelOperator.Batch_Size.value))
        self.window_edit = QLineEdit(self)
        self.window_edit.setText(str(topLevelOperator.Window_Size.value))

        block_label = QLabel("Block size [pixels]:", self)
        block_label.setToolTip("Max size of image blocks sent by Ilastik to the classifier.")
        # Max size of image block (width and height) sent to the classifier for prediction. This will be a large value
        # (typically larger than expected image sizes) because we want the neural network code, not Ilastik to split it into smaller blocks.
        # The only advantage of chosing a smaller block size is that predictions are returned sooner to Ilastik, so it
        # can show prediction progress more frequently. Note that the will be edge artefacts along the edges of these blocks,
        # and that these artefacts are not smoothened at all, since those blocks are not averaged. (Because splitting in
        # blocks happens in Ilastik, and averaging predictions happens in the neuralnets library itself.)

        batch_label = QLabel("Batch size [slices]:", self)
        batch_label.setToolTip("Batch size for the neural network")

        window_label = QLabel("Window size [pixels]:", self)
        window_label.setToolTip("Window size used by neural network when splitting an image into smaller, overlapping blocks.")
        # The 'window size' is used by the neural network code to split a larger image into smaller (and overlapping) blocks.
        # Each block will be passed through the neural network. The predictions along the window edges are not so accurate however, so
        # the overlapping parts of the predictions are averaged to smoothen artefacts along the window edges somewhat.
        # On systems with enough GPU memory this should be set to a large value (e.g. 768 pixels) for most accurate results.
        # If less GPU memory is present, a smaller window size will need to be picked.

        layout = QVBoxLayout()
        layout.addWidget(block_label)
        layout.addWidget(self.block_edit)
        layout.addWidget(batch_label)
        layout.addWidget(self.batch_edit)
        layout.addWidget(window_label)
        layout.addWidget(self.window_edit)
        layout.addWidget(buttonbox)

        self.setLayout(layout)
        self.setWindowTitle("Set Parameters")

    def add_Parameters(self):
        try:
            block_size = int(self.block_edit.text())
        except ValueError:
            block_size = 16 * 1024

        try:
            batch_size = int(self.batch_edit.text())
        except ValueError:
            batch_size = 1

        try:
            window_size = int(self.window_edit.text())
        except ValueError:
            window_size = 256

        self.topLevelOperator.Block_Size.setValue(block_size)
        self.topLevelOperator.Batch_Size.setValue(batch_size)
        self.topLevelOperator.Window_Size.setValue(window_size)

        # close dialog
        super(ParameterDlg, self).accept()


# class SavingDlg(QDialog):
#     """
#     Saving Option Dialog
#     """
#
#     def __init__(self, topLevelOperator, parent):
#         super(QDialog, self).__init__(parent=parent)
#
#         self.topLevelOperator = topLevelOperator
#
#         buttonbox = QDialogButtonBox(Qt.Horizontal, parent=self)
#         buttonbox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
#         buttonbox.accepted.connect(self.change_state)
#         buttonbox.rejected.connect(self.reject)
#
#         self.checkbox = QCheckBox()
#         self.checkbox.setChecked(self.topLevelOperator.SaveFullModel.value)
#         self.checkbox.setCheckable(True)
#         self.checkbox.setText("Enable Model Object serialization")
#
#         layout = QVBoxLayout()
#         layout.addWidget(self.checkbox)
#         layout.addWidget(buttonbox)
#
#         self.setLayout(layout)
#         self.setWindowTitle("Saving Options")
#
#     def change_state(self):
#
#         self.topLevelOperator.SaveFullModel.setValue(self.checkbox.isChecked())
#
#         # close dialog
#         super(SavingDlg, self).accept()


class DLClassGui(LayerViewerGui):
    """
    LayerViewerGui class for Neural Network Classification
    """

    def viewerControlWidget(self):
        """
        Return the widget that controls how the content of the central widget is displayed
        """
        return self._viewerControlUi

    def centralWidget(self):
        """
        Return the widget that will be displayed in the main viewer area.
        """
        return self

    def stopAndCleanUp(self):
        """
        The gui should stop updating all data views and should clean up any resources it created
        """
        for fn in self.__cleanup_fns:
            fn()

    def menus(self):
        """
        Return a list of QMenu widgets to be shown in the menu bar when this applet is visible
        """
        menus = super(DLClassGui, self).menus()

        advanced_menu = QMenu("Advanced", parent=self)

        def settingParameter():
            """
            changing BatchSize, HaloSize and WindowSize parameters
            """
            dlg = ParameterDlg(self.topLevelOperator, parent=self)
            dlg.exec_()

            self.block_size = self.topLevelOperator.Block_Size.value
            self.batch_size = self.topLevelOperator.Batch_Size.value
            self.window_size = self.topLevelOperator.Window_Size.value

        set_parameter = advanced_menu.addAction("Parameters...")
        set_parameter.triggered.connect(settingParameter)

        # def serializing_options():
        #     """
        #     enable/disable serialization options
        #     """
        #     dlg = SavingDlg(self.topLevelOperator, parent=self)
        #     dlg.exec_()
        #
        #     if self.topLevelOperator.SaveFullModel.value == True:
        #         obj_list = []
        #         for key in self.topLevelOperator.ModelPath.value:  # FIXME: meanwhile ModelPath is just a single path string, not a list/dict
        #             logger.debug(f"dlClassGui serializing_options(): loading neural net {self.topLevelOperator.ModelPath.value[key]}")
        #             object_ = load_net(self.topLevelOperator.ModelPath.value[key])
        #             obj_list.append(object_)
        #
        #         self.topLevelOperator.FullModel.setValue(obj_list)

        # advanced_menu.addAction("Saving Options").triggered.connect(serializing_options)

        menus += [advanced_menu]

        return menus

    def appletDrawer(self):
        """
        Return the drawer widget for this applet
        """
        return self.drawer

    def __init__(self, parentApplet, topLevelOperator):
        super(DLClassGui, self).__init__(parentApplet, topLevelOperator)

        self.parentApplet = parentApplet
        self.drawer = None
        self.topLevelOperator = topLevelOperator
        self.classifier = None

        self.__cleanup_fns = []

        self._initAppletDrawerUic()
        self.initViewerControls()
        self.initViewerControlUi()

        self.batch_size = self.topLevelOperator.Batch_Size.value
        self.block_size = self.topLevelOperator.Block_Size.value
        self.window_size = self.topLevelOperator.Window_Size.value

    def _initAppletDrawerUic(self, drawerPath=None):
        """
        Load the ui file for the applet drawer, which we own.
        """
        if drawerPath is None:
            localDir = os.path.split(__file__)[0]
            drawerPath = os.path.join(localDir, "dlClassificationDrawer.ui")
        self.drawer = uic.loadUi(drawerPath)

        self._setModelGuiVisible(False)
        self.drawer.comboBox.clear()
        self.drawer.liveUpdateButton.clicked.connect(self._livePredictionClicked)
        self.drawer.loadModel.clicked.connect(self._loadModelButtonClicked)

        if self.topLevelOperator.ModelPath.ready():
            self.classifier = self.topLevelOperator.ModelPath.value
            self.drawer.comboBox.clear()
            self.drawer.comboBox.addItem(self.topLevelOperator.ModelPath.value)

        # Model for our list of classification labels (=classes)
        model = LabelListModel()
        self.drawer.labelListView.setModel(model)
        self.drawer.labelListModel = model

        # Add fixed labels for two classes. Currently our neural networks discriminate between two classes only.
        self._addNewLabel("Background", self._colorForLabel(1), None, makePermanent=True)
        self._addNewLabel("Foreground", self._colorForLabel(2), None, makePermanent=True)

    def _colorForLabel(self, n):
        # Note: entry 0 in the colortable is transparent
        color = QColor()
        color.setRgba(colortables.default16_new[n])
        return color

    def _addNewLabel(self, labelName, labelColor, pmapColor, makePermanent=True):
        """
        Add a new label to the label list GUI control.
        (Note: In the GUI, the color patch to the left of a label consists of two halves whose colors can be changed
        independently. The top left half is the color used for painting labels interactively; the bottom right half
        for drawing in the segmentation and probability layers.)
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

        label = Label(labelName, labelColor, pmapColor=pmapColor)

        # Insert new label
        newRow = self.drawer.labelListModel.rowCount()
        self.drawer.labelListModel.insertRow(newRow, label)

        if makePermanent:
            self.drawer.labelListModel.makeRowPermanent(newRow)

        # Make the new label selected
        selectedRow = self.drawer.labelListModel.rowCount() - 1
        self.drawer.labelListModel.select(selectedRow)

        QApplication.restoreOverrideCursor()

    def initViewerControls(self):
        """
        initializing viewerControl
        """
        self._viewerControlWidgetStack = QStackedWidget(parent=self)

    def initViewerControlUi(self):
        """
        Load the viewer controls GUI, which appears below the applet bar.
        In our case, the viewer control GUI consists of the "Group Visibility" toggles "Probability" and "Segmentation"
        with the list over layers underneath.
        """
        localDir = os.path.split(__file__)[0]
        self._viewerControlUi = uic.loadUi(os.path.join(localDir, "viewerControls.ui"))

        def nextCheckState(checkbox):
            """
            sets the checkbox to the next state
            """
            checkbox.setChecked(not checkbox.isChecked())

        self._viewerControlUi.checkShowPredictions.nextCheckState = partial(
            nextCheckState, self._viewerControlUi.checkShowPredictions
        )
        self._viewerControlUi.checkShowSegmentation.nextCheckState = partial(
            nextCheckState, self._viewerControlUi.checkShowSegmentation
        )

        self._viewerControlUi.checkShowPredictions.clicked.connect(self.handleShowPredictionsClicked)
        self._viewerControlUi.checkShowSegmentation.clicked.connect(self.handleShowSegmentationClicked)

        model = self.editor.layerStack
        self._viewerControlUi.viewerControls.setupConnections(model)

    def setupLayers(self):
        """
        which layers will be shown in the layerviewergui.
        Triggers the prediction by setting the layer on visible
        """

        inputSlot = self.topLevelOperator.InputImage

        layers = []

        labels = self.drawer.labelListModel

        # Add the segmentations
        for channel, segmentationSlot in enumerate(self.topLevelOperatorView.SegmentationChannels):
            if segmentationSlot.ready():
                ref_label = labels[channel]
                segsrc = createDataSource(segmentationSlot)
                segLayer = AlphaModulatedLayer(
                    segsrc, tintColor=ref_label.pmapColor(), range=(0.0, 1.0), normalize=(0.0, 1.0)
                )

                segLayer.opacity = 1
                segLayer.visible = False
                segLayer.visibleChanged.connect(self.updateShowSegmentationCheckbox)

                def setLayerColor(c, segLayer_=segLayer, initializing=False):
                    if not initializing and segLayer_ not in self.layerstack:
                        # This layer has been removed from the layerstack already.
                        # Don't touch it.
                        return
                    segLayer_.tintColor = c

                def setSegLayerName(n, segLayer_=segLayer, initializing=False):
                    if not initializing and segLayer_ not in self.layerstack:
                        # This layer has been removed from the layerstack already.
                        # Don't touch it.
                        return
                    newName = "Segmentation of %s" % n
                    segLayer_.name = newName

                setSegLayerName(ref_label.name, initializing=True)
                ref_label.pmapColorChanged.connect(setLayerColor)
                ref_label.nameChanged.connect(setSegLayerName)
                layers.append(segLayer)

        # Add the prediction probabilities
        for channel, predictionSlot in enumerate(self.topLevelOperator.PredictionProbabilityChannels):
            if predictionSlot.ready():
                ref_label = labels[channel]
                predictsrc = createDataSource(predictionSlot)
                predictionLayer = AlphaModulatedLayer(predictsrc, tintColor=ref_label.pmapColor(),
                                                      range=(0.0, 1.0), normalize=(0.0, 1.0))
                predictionLayer.opacity = 0.25
                predictionLayer.visible = self.drawer.liveUpdateButton.isChecked()
                predictionLayer.visibleChanged.connect(self.updateShowPredictionCheckbox)

                def setLayerColor(c, predictLayer_=predictionLayer, initializing=False):
                    if not initializing and predictLayer_ not in self.layerstack:
                        # This layer has been removed from the layerstack already.
                        # Don't touch it.
                        return
                    predictLayer_.tintColor = c

                def setPredLayerName(n, predictLayer_=predictionLayer, initializing=False):
                    """
                    function for setting the names for every Channel
                    """
                    if not initializing and predictLayer_ not in self.layerstack:
                        # This layer has been removed from the layerstack already.
                        # Don't touch it.
                        return
                    newName = "Probability of %s" % n
                    predictLayer_.name = newName

                setPredLayerName(ref_label.name, initializing=True)
                ref_label.pmapColorChanged.connect(setLayerColor)
                ref_label.nameChanged.connect(setPredLayerName)
                layers.append(predictionLayer)

        # The raw input data, always as last layer
        if inputSlot.ready():
            rawLayer = self.createStandardLayerFromSlot(inputSlot)
            rawLayer.visible = True
            rawLayer.opacity = 1.0
            rawLayer.name = "Raw Data (display only)"
            layers.append(rawLayer)

        return layers

    def _setClassifierModel(self, filename):
        self.classifier = filename

        # Create combo box with just a single model / classifier.
        # Perhaps later we ought to use something else than a combo box.
        modelname = os.path.basename(os.path.normpath(filename))
        self.drawer.comboBox.clear()
        self.drawer.comboBox.addItem(modelname)

        # Create neural network classifier object
        # We do not want Ilastik's halo's. the neuralnet does its own overlapping and averaging, so we want to send
        # a full image to the neural net - so we pick a very large block size here; so large that the full image fits
        # in it and ilastik only does blocking on the z-planes, but not in x and y.
        model = DeepLearningLazyflowClassifier(None, filename, self.batch_size, self.window_size)

        block_shape = numpy.array([self.batch_size, self.block_size, self.block_size,
                                   None])  # (batch size, height, width, ???)   CHECKME: what is the None for?
        # block_shape is the size of the images that ilastik will feed to the classifier for prediction
        logger.debug(f"Using block_shape {block_shape}")

        self.topLevelOperator.ModelPath.setValue(self.classifier)
        self.topLevelOperator.FreezePredictions.setValue(True)
        self.topLevelOperator.BlockShape.setValue(block_shape)
        self.topLevelOperator.NumClasses.setValue(model._net.out_channels)
        self.topLevelOperator.Classifier.setValue(model)

        self.updateAllLayers()  # CHECKME needed?
        self.parentApplet.appletStateUpdateRequested()  # CHECKME needed?

    def _livePredictionClicked(self):
        """
        """
        livePrediction = self.drawer.liveUpdateButton.isChecked()
        self.topLevelOperator.FreezePredictions.setValue(not livePrediction)
        if livePrediction:
            self.updateAllLayers()
        self.parentApplet.appletStateUpdateRequested()

    @pyqtSlot()
    def handleShowPredictionsClicked(self):
        """
        sets the layer visibility when showPrediction is clicked
        """
        checked = self._viewerControlUi.checkShowPredictions.isChecked()
        for layer in self.layerstack:
            if "Probability" in layer.name:
                layer.visible = checked

    @pyqtSlot()
    def handleShowSegmentationClicked(self):
        checked = self._viewerControlUi.checkShowSegmentation.isChecked()
        for layer in self.layerstack:
            if "Segmentation" in layer.name:
                layer.visible = checked

    @pyqtSlot()
    def updateShowPredictionCheckbox(self):
        """
        updates the "Probability" checkbox, when predictions were added to the layers
        """
        predictLayerCount = 0
        visibleCount = 0
        for layer in self.layerstack:
            if "Probability" in layer.name:
                predictLayerCount += 1
                if layer.visible:
                    visibleCount += 1

        if visibleCount == 0:
            self._viewerControlUi.checkShowPredictions.setCheckState(Qt.Unchecked)
        elif predictLayerCount == visibleCount:
            self._viewerControlUi.checkShowPredictions.setCheckState(Qt.Checked)
        else:
            self._viewerControlUi.checkShowPredictions.setCheckState(Qt.PartiallyChecked)

    @pyqtSlot()
    def updateShowSegmentationCheckbox(self):
        """
        updates the "Segmentation" checkbox, when segmentations were added to the layers
        """
        segLayerCount = 0
        visibleCount = 0
        for layer in self.layerstack:
            if "Segmentation" in layer.name:
                segLayerCount += 1
                if layer.visible:
                    visibleCount += 1

        if visibleCount == 0:
            self._viewerControlUi.checkShowSegmentation.setCheckState(Qt.Unchecked)
        elif segLayerCount == visibleCount:
            self._viewerControlUi.checkShowSegmentation.setCheckState(Qt.Checked)
        else:
            self._viewerControlUi.checkShowSegmentation.setCheckState(Qt.PartiallyChecked)

    def _setModelGuiVisible(self, enable):
        """
        Show/hide the part of the user interface with the combo box with loaded models,
        the list with labels and the Live Prediction button.
        """
        self.drawer.modelsLabel.setVisible(enable)
        self.drawer.comboBox.setVisible(enable)
        self.drawer.labelsLabel.setVisible(enable)
        self.drawer.labelListView.setVisible(enable)
        self.drawer.liveUpdateButton.setVisible(enable)

    def _loadModelButtonClicked(self):
        """
        When Load Model button is clicked.
        """
        mostRecentModelFile = preferences.get("DataSelection", "recent neural net")
        if mostRecentModelFile is not None:
            defaultDirectory = os.path.split(mostRecentModelFile)[0]
        else:
            defaultDirectory = os.path.expanduser("~")

        fileName = self.getModelFileNameToOpen(self, defaultDirectory)

        if fileName is not None:
            self._setClassifierModel(fileName)
            preferences.set("DataSelection", "recent neural net", fileName)
            self._setModelGuiVisible(True)

    def getModelFileNameToOpen(cls, parent_window, defaultDirectory):
        """
        opens a QFileDialog for importing files
        """
        extensions = ["pytorch"]
        filter_strs = ["*." + x for x in extensions]
        filters = ["{filt} ({filt})".format(filt=x) for x in filter_strs]
        filt_all_str = "Neural nets (" + " ".join(filter_strs) + ")"

        fileName = None

        if ilastik_config.getboolean("ilastik", "debug"):
            # use Qt dialog in debug mode (more portable?)
            file_dialog = QFileDialog(parent_window, "Select Model")
            file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
            # do not display file types associated with a filter
            # the line for "Image files" is too long otherwise
            file_dialog.setNameFilters([filt_all_str] + filters)
            # file_dialog.setNameFilterDetailsVisible(False)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setDirectory(defaultDirectory)

            if file_dialog.exec_():
                fileName = file_dialog.selectedFiles()[0]
        else:
            # otherwise, use native dialog of the present platform
            fileName, _ = QFileDialog.getOpenFileName(parent_window, "Select Model", defaultDirectory, filt_all_str)

        return fileName
