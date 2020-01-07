###############################################################################
# VIB, Frank Vernaillen, Jan 2020
# Implementation based on ilastik\ilastik\applets\networkClassification
###############################################################################

from ilastik.applets.base.appletSerializer import AppletSerializer, SerialListSlot, SerialDictSlot, SerialPickleableSlot

import logging

logger = logging.getLogger(__name__)


class DLClassificationSerializer(AppletSerializer):
    def __init__(self, topLevelOperator, projectFileGroupName):
        self.VERSION = 1

        slots = [
            SerialPickleableSlot(topLevelOperator.FullModel, version=1),
            SerialDictSlot(topLevelOperator.ModelPath),
        ]

        super(DLClassificationSerializer, self).__init__(projectFileGroupName, slots)
