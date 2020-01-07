###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################

from __future__ import absolute_import
from ilastik.applets.dataExport.dataExportApplet import DataExportApplet


class DLClassificationDataExportApplet(DataExportApplet):
    """
    This a specialization of the generic data export applet that
    provides a special viewer for Neural Network predictions.
    """

    def __init__(self, workflow, title, isBatch=False):
        # Base class init
        super(DLClassificationDataExportApplet, self).__init__(workflow, title, isBatch)

    def getMultiLaneGui(self):
        if self._gui is None:
            # Gui is a special subclass of the generic gui
            from .dlClassificationDataExportGui import DLClassificationDataExportGui

            self._gui = DLClassificationDataExportGui(self, self.topLevelOperator)
        return self._gui
