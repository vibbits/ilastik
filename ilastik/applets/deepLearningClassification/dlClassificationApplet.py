###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################

from __future__ import absolute_import
from ilastik.applets.base.standardApplet import StandardApplet
from .opDLClassification import OpDLClassification
from .dlClassificationSerializer import DLClassificationSerializer


class DLClassificationApplet(StandardApplet):
    """
    StandardApplet Subclass with SingleLaneGui and SingeLaneOperator
    """

    def __init__(self, workflow, projectFileGroupName):

        super(DLClassificationApplet, self).__init__("VIB Deep Learning Classification", workflow=workflow)

        self._serializableItems = [
            DLClassificationSerializer(self.topLevelOperator, projectFileGroupName)
        ]  # Legacy (v0.5) importer
        self._gui = None
        self.predictionSerializer = self._serializableItems[0]

    @property
    def broadcastingSlots(self):
        """
        defines which variables will be shared with different lanes
        """
        return ["ModelPath", "FreezePredictions"]  # perhaps lateron "FullModel" too

    @property
    def dataSerializers(self):
        """
        A list of dataSerializer objects for loading/saving any project data the applet is responsible for
        """
        return self._serializableItems

    @property
    def singleLaneGuiClass(self):
        """
        This applet uses a single lane gui and shares variables through the broadcasting slots
        """
        from .dlClassificationGui import DLClassGui

        return DLClassGui

    @property
    def singleLaneOperatorClass(self):
        """
        Return the operator class which handles a single image.
        """
        return OpDLClassification
