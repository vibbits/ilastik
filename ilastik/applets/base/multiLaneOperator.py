from abc import ABCMeta, abstractmethod
from lazyflow.graph import OperatorWrapper
from ilastik.utility.operatorSubView import OperatorSubView

def _has_attribute( cls, attr ):
    return True if any(attr in B.__dict__ for B in cls.__mro__) else False

class MultiLaneOperatorABC(object):
    """
    This abstract base class specifies the interface to which all top-level applet operators must adhere.
    The distinguishing characteristic of a top-level operator is the fact that they must be capable of 
    supporting multiple images via multi-slots that are indexed by image lane number.
    
    Image lanes of the top-level operator are added, removed, and accessed via the ``addLane``, ``removeLane``, and ``getLane`` functions.
    
    Note: Most applets can simply inherit from the ``StandardApplet`` base class, 
    which will automatically adapt single-lane top-level operators to satisfy this interface. 
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def addLane(self, laneIndex):
        """
        Add an image lane.
        """
        raise NotImplementedError

    @abstractmethod
    def removeLane(self, laneIndex, finalLength):
        """
        Remove an image lane.
        """
        raise NotImplementedError

    @abstractmethod
    def getLane(self, laneIndex):
        """
        Get an object that exposes the relevant slots for the specific lane.
        The object may be an operator, or may merely be an operator-like "view" object.
        """
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, C):
        """
        This function allows us to compare objects to the MultiLaneOperator interface
        even if they don't happen to inherit from this base class.
        """
        if cls is MultiLaneOperatorABC:
            retval = True
            retval &= _has_attribute( cls, 'addLane' )
            retval &= _has_attribute( cls, 'removeLane' )
            retval &= _has_attribute( cls, 'getLane' )
            return retval
        return NotImplemented

class OpAutoMultiLane( OperatorWrapper ):
    """
    An extension of the OperatorWrapper that provides the functions needed to satisfy MultiLaneOperatorABC.
    """
    
    def addLane(self, laneIndex):
        """
        Add an image lane.
        """
        numLanes = len(self.innerOperators)
        assert numLanes == laneIndex, "Image lanes must be appended."        
        self._insertInnerOperator(numLanes, numLanes+1)

    def removeLane(self, laneIndex, finalLength):
        """
        Remove an image lane.
        """
        numLanes = len(self.innerOperators)
        self._removeInnerOperator(laneIndex, numLanes-1)

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)
