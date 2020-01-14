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

from volumina.api import createDataSource, AlphaModulatedLayer
from volumina.utility import preferences

from PyQt5 import uic
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
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


class SavingDlg(QDialog):
    """
    Saving Option Dialog
    """

    def __init__(self, topLevelOperator, parent):
        super(QDialog, self).__init__(parent=parent)

        self.topLevelOperator = topLevelOperator

        buttonbox = QDialogButtonBox(Qt.Horizontal, parent=self)
        buttonbox.setStandardButtons(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonbox.accepted.connect(self.change_state)
        buttonbox.rejected.connect(self.reject)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(self.topLevelOperator.SaveFullModel.value)
        self.checkbox.setCheckable(True)
        self.checkbox.setText("Enable Model Object serialization")

        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addWidget(buttonbox)

        self.setLayout(layout)
        self.setWindowTitle("Saving Options")

    def change_state(self):

        self.topLevelOperator.SaveFullModel.setValue(self.checkbox.isChecked())

        # close dialog
        super(SavingDlg, self).accept()


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
            changing BatchSize and HaloSize
            """
            dlg = ParameterDlg(self.topLevelOperator, parent=self)
            dlg.exec_()

            self.block_size = self.topLevelOperator.Block_Size.value
            self.batch_size = self.topLevelOperator.Batch_Size.value
            self.window_size = self.topLevelOperator.Window_Size.value

        set_parameter = advanced_menu.addAction("Parameters...")
        set_parameter.triggered.connect(settingParameter)

        def serializing_options():
            """
            enable/disable serialization options
            """
            dlg = SavingDlg(self.topLevelOperator, parent=self)
            dlg.exec_()

            if self.topLevelOperator.SaveFullModel.value == True:
                obj_list = []
                # print(list(self.topLevelOperator.ModelPath.value.values())[0])
                # object_ = torch.load(list(self.topLevelOperator.ModelPath.value.values())[0])
                for key in self.topLevelOperator.ModelPath.value:
                    logger.debug(f"dlClassGui serializing_options(): loading neural net {self.topLevelOperator.ModelPath.value[key]}")
                    object_ = load_net(self.topLevelOperator.ModelPath.value[key])
                    obj_list.append(object_)

                self.topLevelOperator.FullModel.setValue(obj_list)

        advanced_menu.addAction("Saving Options").triggered.connect(serializing_options)

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
        self.classifiers = OrderedDict()

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
            drawerPath = os.path.join(localDir, "dlClassAppletUiTest.ui")
        self.drawer = uic.loadUi(drawerPath)

        self.drawer.comboBox.clear()
        self.drawer.liveUpdateButton.clicked.connect(self.dlPredict)
        self.drawer.addModel.clicked.connect(self.addModel)

        if self.topLevelOperator.ModelPath.ready():

            self.drawer.comboBox.clear()
            self.drawer.comboBox.addItems(self.topLevelOperator.ModelPath.value)

            self.classifiers = self.topLevelOperator.ModelPath.value

    def initViewerControls(self):
        """
        initializing viewerControl
        """
        self._viewerControlWidgetStack = QStackedWidget(parent=self)

    def initViewerControlUi(self):
        """
        Load the viewer controls GUI, which appears below the applet bar.
        In our case, the viewer control GUI consists mainly of a layer list.
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

        self._viewerControlUi.checkShowPredictions.clicked.connect(self.handleShowPredictionsClicked)

        model = self.editor.layerStack
        self._viewerControlUi.viewerControls.setupConnections(model)

    def setupLayers(self):
        """
        which layers will be shown in the layerviewergui.
        Triggers the prediction by setting the layer on visible
        """

        inputSlot = self.topLevelOperator.InputImage

        layers = []

        tintColors = [QColor(0, 0, 255), QColor(255, 0, 0)]  # a couple of predefined class label colors

        for channel, predictionSlot in enumerate(self.topLevelOperator.PredictionProbabilityChannels):
            if predictionSlot.ready():
                predictsrc = createDataSource(predictionSlot)
                predictionLayer = AlphaModulatedLayer(predictsrc, tintColor=tintColors[channel % len(tintColors)],
                                                      range=(0.0, 1.0), normalize=(0.0, 1.0))
                predictionLayer.visible = self.drawer.liveUpdateButton.isChecked()
                predictionLayer.opacity = 0.25
                predictionLayer.visibleChanged.connect(self.updateShowPredictionCheckbox)

                def setPredLayerName(n, predictLayer_=predictionLayer, initializing=False):
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

                layers.append(predictionLayer)

        # always as last layer
        if inputSlot.ready():
            rawLayer = self.createStandardLayerFromSlot(inputSlot)
            rawLayer.visible = True
            rawLayer.opacity = 1.0
            rawLayer.name = "Raw Data (display only)"
            layers.append(rawLayer)

        return layers

    def add_DL_classifier(self, filename):
        """
        Adds the chosen FilePath to the classifierDictionary and to the ComboBox
        """

        # split path string
        modelname = os.path.basename(os.path.normpath(filename))

        # Statement for importing the same classifier twice
        if modelname in self.classifiers.keys():
            print("Classifier already added")
            QMessageBox.critical(self, "Error loading file", "{} already added".format(modelname))
        else:

            # serialization problems because of group names when using the classifier function as value
            # self.classifiers[modelname] = TikTorchLazyflowClassifier(None, filename[0], halo_size, batch_size)

            # workAround
            self.classifiers[modelname] = filename

            # clear first the comboBox or addItems will duplicate names
            self.drawer.comboBox.clear()
            self.drawer.comboBox.addItems(self.classifiers)

            if self.topLevelOperator.SaveFullModel.value == True:
                logger.debug(f"dlClassGui add_DL_classifiers(): loading {filename}")
                object_ = load_net(filename)
                self.topLevelOperator.FullModel.setValue(object_)

            else:
                self.topLevelOperator.ModelPath.setValue(self.classifiers)

    def dlPredict(self):
        """
        When LivePredictionButton is clicked.
        Sets the ClassifierSlotValue for Prediction.
        Updates the SetupLayers function
        """
        classifier_key = self.drawer.comboBox.currentText()
        classifier_index = self.drawer.comboBox.currentIndex()

        if len(classifier_key) == 0:
            QMessageBox.critical(self, "Error loading file", "Add a Model first")

        else:
            if self.drawer.liveUpdateButton.isChecked():

                if self.topLevelOperator.FullModel.value:
                    # if the full model object is serialized
                    model_object = self.topLevelOperator.FullModel.value[classifier_index]
                    print(classifier_index)
                    model_path = None
                else:
                    model_object = None
                    model_path = self.classifiers[classifier_key]

                self.topLevelOperator.FreezePredictions.setValue(False)

                # Create neural network classifier object
                # we do not want halo's. the neuralnet does its own overlapping and averaging, so we want to send a full image to the neural net - so we pick a very large block size here; so large that the full image fits in it and ilastik only does blocking on the z-planes, but not in x and y
                model = DeepLearningLazyflowClassifier(model_object, model_path, self.batch_size, self.window_size)

                block_shape = numpy.array([self.batch_size, self.block_size, self.block_size, None])  # (batch size, height, width, ???)   CHECKME: what is the None for?
                # block_shape is the size of the images that ilastik will feed to the classifier for prediction
                logger.debug(f"Using block_shape {block_shape}")

                self.topLevelOperator.BlockShape.setValue(block_shape)
                self.topLevelOperator.NumClasses.setValue(model._net.out_channels)

                self.topLevelOperator.Classifier.setValue(model)

                self.updateAllLayers()
                self.parentApplet.appletStateUpdateRequested()

            else:
                # when disabled, the user can scroll around without predicting
                self.topLevelOperator.FreezePredictions.setValue(True)
                self.parentApplet.appletStateUpdateRequested()

    @pyqtSlot()
    def handleShowPredictionsClicked(self):
        """
        sets the layer visibility when showPredicition is clicked
        """
        checked = self._viewerControlUi.checkShowPredictions.isChecked()
        for layer in self.layerstack:
            if "Prediction" in layer.name:
                layer.visible = checked

    @pyqtSlot()
    def updateShowPredictionCheckbox(self):
        """
        updates the showPrediction Checkbox when Predictions were added to the layers
        """
        predictLayerCount = 0
        visibleCount = 0
        for layer in self.layerstack:
            if "Prediction" in layer.name:
                predictLayerCount += 1
                if layer.visible:
                    visibleCount += 1

        if visibleCount == 0:
            self._viewerControlUi.checkShowPredictions.setCheckState(Qt.Unchecked)
        elif predictLayerCount == visibleCount:
            self._viewerControlUi.checkShowPredictions.setCheckState(Qt.Checked)
        else:
            self._viewerControlUi.checkShowPredictions.setCheckState(Qt.PartiallyChecked)

    def addModel(self):
        """
        When Add Model button is clicked.
        """
        mostRecentModelFile = preferences.get("DataSelection", "recent neural net")
        if mostRecentModelFile is not None:
            defaultDirectory = os.path.split(mostRecentModelFile)[0]
        else:
            defaultDirectory = os.path.expanduser("~")

        fileName = self.getModelFileNameToOpen(self, defaultDirectory)

        if fileName is not None:
            self.add_DL_classifier(fileName)
            preferences.set("DataSelection", "recent neural net", fileName)

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
