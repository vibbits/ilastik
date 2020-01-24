###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based (heavily) on Ilastik's nnClassificationWorkflow.py
###############################################################################
from __future__ import division
import argparse
import logging

from ilastik.workflow import Workflow
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.deepLearningClassification import DLClassificationApplet, DLClassificationDataExportApplet
from ilastik.applets.batchProcessing import BatchProcessingApplet

from lazyflow.graph import Graph

logger = logging.getLogger(__name__)


class DLClassificationWorkflow(Workflow):
    """
    Workflow for the VIB Deep Learning Classification Applet
    """

    workflowName = "VIB Deep Learning Classification"
    workflowDescription = "VIB's deep neural net for pixel classification.."
    defaultAppletIndex = 0  # show DataSelection by default

    DATA_ROLE_RAW = 0
    ROLE_NAMES = ["Raw Data"]
    EXPORT_NAMES = ["Probabilities"]  # CHECKME: should we add segmentations too?

    @property
    def applets(self):
        """
        Return the list of applets that are owned by this workflow
        """
        return self._applets

    @property
    def imageNameListSlot(self):
        """
        Return the "image name list" slot, which lists the names of
        all image lanes (i.e. files) currently loaded by the workflow
        """
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def __init__(self, shell, headless, workflow_cmdline_args, project_creation_args, *args, **kwargs):

        # Create a graph to be shared by all operators
        graph = Graph()
        super(DLClassificationWorkflow, self).__init__(
            shell, headless, workflow_cmdline_args, project_creation_args, graph=graph, *args, **kwargs
        )
        self._applets = []
        self._workflow_cmdline_args = workflow_cmdline_args
        # Parse workflow-specific command-line args
        parser = argparse.ArgumentParser()
        # parser.add_argument('--print-labels-by-slice', help="Print the number of labels for each Z-slice of each image.", action="store_true")

        # Parse the creation args: These were saved to the project file when this project was first created.
        parsed_creation_args, unused_args = parser.parse_known_args(project_creation_args)

        # Parse the cmdline args for the current session.
        parsed_args, unused_args = parser.parse_known_args(workflow_cmdline_args)
        # self.print_labels_by_slice = parsed_args.print_labels_by_slice

        data_instructions = (
            "Select your input data using the 'Raw Data' tab shown on the right.\n\n"
            "Power users: Optionally use the 'Prediction Mask' tab to supply a binary image that tells ilastik where it should avoid computations you don't need."
        )

        # Applets for training (interactive) workflow
        self.dataSelectionApplet = self.createDataSelectionApplet()
        opDataSelection = self.dataSelectionApplet.topLevelOperator

        # see role constants, above
        opDataSelection.DatasetRoles.setValue(DLClassificationWorkflow.ROLE_NAMES)

        self.dlClassificationApplet = DLClassificationApplet(self, "DLClassificationApplet")

        self.dataExportApplet = DLClassificationDataExportApplet(self, "Prediction Export")
        self.dataExportApplet.prepare_for_entire_export = self.prepare_for_entire_export
        self.dataExportApplet.post_process_entire_export = self.post_process_entire_export

        # Configure global DataExport settings
        opDataExport = self.dataExportApplet.topLevelOperator
        opDataExport.WorkingDirectory.connect(opDataSelection.WorkingDirectory)
        opDataExport.SelectionNames.setValue(self.EXPORT_NAMES)

        self.batchProcessingApplet = BatchProcessingApplet(
            self, "Batch Processing", self.dataSelectionApplet, self.dataExportApplet
        )

        # Expose for shell
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.dlClassificationApplet)
        self._applets.append(self.dataExportApplet)
        self._applets.append(self.batchProcessingApplet)

        if unused_args:
            # We parse the export setting args first.  All remaining args are considered input files by the input applet.
            self._batch_export_args, unused_args = self.dataExportApplet.parse_known_cmdline_args(unused_args)
            self._batch_input_args, unused_args = self.batchProcessingApplet.parse_known_cmdline_args(unused_args)
        else:
            self._batch_input_args = None
            self._batch_export_args = None

        if unused_args:
            logger.warn("Unused command-line args: {}".format(unused_args))

    def createDataSelectionApplet(self):
        """
        Can be overridden by subclasses, if they want to use
        special parameters to initialize the DataSelectionApplet.
        """
        data_instructions = "Select your input data using the 'Raw Data' tab shown on the right"
        return DataSelectionApplet(
            self, "Input Data", "Input Data", supportIlastik05Import=True, instructionText=data_instructions
        )

    def connectLane(self, laneIndex):
        """
        connects the operators for different lanes, each lane has a laneIndex starting at 0
        """
        opData = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opDLclassify = self.dlClassificationApplet.topLevelOperator.getLane(laneIndex)
        opDataExport = self.dataExportApplet.topLevelOperator.getLane(laneIndex)

        # Input Image -> Classification Op
        opDLclassify.InputImage.connect(opData.Image)
        # opClassify.PredictionMasks.connect(opData.ImageGroup[self.Roles.PREDICTION_MASK])   # Needed??

        # Data Export connections
        opDataExport.RawData.connect(opData.ImageGroup[self.DATA_ROLE_RAW])
        opDataExport.RawDatasetInfo.connect(opData.DatasetGroup[self.DATA_ROLE_RAW])
        opDataExport.Inputs.resize(len(self.EXPORT_NAMES))
        opDataExport.Inputs[0].connect(opDLclassify.CachedPredictionProbabilities)
        for slot in opDataExport.Inputs:
            assert slot.upstream_slot is not None

    def handleAppletStateUpdateRequested(self):
        """
        Overridden from Workflow base class
        Called when an applet has fired the :py:attr:`Applet.appletStateUpdateRequested`
        """
        # If no data, nothing else is ready.
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        input_ready = len(opDataSelection.ImageGroup) > 0 and not self.dataSelectionApplet.busy

        opDLClassification = self.dlClassificationApplet.topLevelOperator

        opDataExport = self.dataExportApplet.topLevelOperator

        predictions_ready = input_ready and len(opDataExport.Inputs) > 0
        # opDataExport.Inputs[0][0].ready()
        # (TinyVector(opDataExport.Inputs[0][0].meta.shape) > 0).all()

        # Problems can occur if the features or input data are changed during live update mode.
        # Don't let the user do that.
        live_update_active = not opDLClassification.FreezePredictions.value

        # The user isn't allowed to touch anything while batch processing is running.
        batch_processing_busy = self.batchProcessingApplet.busy

        logger.info(f"predictions_ready={predictions_ready} input_ready={input_ready} batch_processing_busy={batch_processing_busy} live_update_active={live_update_active}")

        self._shell.setAppletEnabled(self.dataSelectionApplet, not batch_processing_busy)
        self._shell.setAppletEnabled(self.dlClassificationApplet, input_ready and not batch_processing_busy)
        self._shell.setAppletEnabled(
            self.dataExportApplet, predictions_ready and not batch_processing_busy and not live_update_active
        )

        if self.batchProcessingApplet is not None:
            self._shell.setAppletEnabled(self.batchProcessingApplet, predictions_ready and not batch_processing_busy)

        # Lastly, check for certain "busy" conditions, during which we
        # should prevent the shell from closing the project.
        busy = False
        busy |= self.dataSelectionApplet.busy
        busy |= self.dlClassificationApplet.busy
        busy |= self.dataExportApplet.busy
        busy |= self.batchProcessingApplet.busy
        self._shell.enableProjectChanges(not busy)

    def prepare_for_entire_export(self):
        """
        Assigned to DataExportApplet.prepare_for_entire_export
        (See above.)
        """
        # Unfreeze the classifier caches (ensure that we're exporting based on up-to-date labels)
        # CHECKME: Is this really needed? If so, explain in more detail why.
        self.freeze_status = self.dlClassificationApplet.topLevelOperator.FreezePredictions.value
        self.dlClassificationApplet.topLevelOperator.FreezePredictions.setValue(False)

    def post_process_entire_export(self):
        """
        Assigned to DataExportApplet.post_process_entire_export
        (See above.)
        """
        self.dlClassificationApplet.topLevelOperator.FreezePredictions.setValue(self.freeze_status)
