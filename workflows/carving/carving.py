#!/usr/bin/env python

import os, numpy, threading

from PyQt4.QtGui import QShortcut, QKeySequence
from PyQt4.QtGui import QColor, QMenu

from ilastik.workflow import Workflow

from ilastik.applets.projectMetadata import ProjectMetadataApplet
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.layerViewer import LayerViewerApplet
from ilastik.applets.labeling.labelingApplet import LabelingApplet
from ilastik.applets.labeling.labelingGui import LabelingGui
from ilastik.applets.labeling import OpLabeling

from lazyflow.graph import Graph, Operator, OperatorWrapper, InputSlot, OutputSlot
from lazyflow.operators import OpAttributeSelector

from volumina.pixelpipeline.datasources import RelabelingArraySource, LazyflowSource
from volumina.layer import ColortableLayer
from volumina import adaptors

from cylemon.segmentation import MSTSegmentor

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class OpCarvingTopLevel(Operator):
    name = "OpCarvingTopLevel"
    
    RawData = InputSlot(level=1)

    def __init__(self, carvingGraphFile, *args, **kwargs):
        super(OpCarvingTopLevel, self).__init__(*args, **kwargs)
        
        self.opLabeling = OpLabeling(graph=self.graph, parent=self)
        self.opCarving = OperatorWrapper( OpCarving(carvingGraphFile, graph=self.graph) )
        
        self.opLabeling.InputImages.connect( self.RawData )
        self.opCarving.RawData.connect( self.RawData )
        
        self.opCarving.WriteSeeds.connect(self.opLabeling.LabelInputs)

    def saveCurrentObject(self, imageIndex):  
        self.opCarving.innerOperators[imageIndex].saveCurrentObject()
        # Sparse label array automatically shifts label values down 1
        self.opLabeling.LabelDelete.setValue(2)
        self.opLabeling.LabelDelete.setValue(1)
        self.opLabeling.LabelDelete.setValue(-1)
        # trigger a re-computation
        self.opCarving.innerOperators[imageIndex].Trigger.setDirty(slice(None))
        
    def saveObjectAs(self, name, imageIndex):
        # first, save the object under "name"
        self.opCarving.innerOperators[imageIndex].saveObjectAs(name)
        # Sparse label array automatically shifts label values down 1
        
        nonzeroSlicings = self.opLabeling.NonzeroLabelBlocks[imageIndex][:].wait()[0]
      
        def coordinateList(): 
            coors1 = [[], [], []]
            coors2 = [[], [], []]
            for sl in nonzeroSlicings:
                a = self.opLabeling.LabelImages[imageIndex][sl].wait()
                w1 = numpy.where(a == 1)
                w2 = numpy.where(a == 2)
                w1 = [w1[i] + sl[i].start for i in range(3)]
                w2 = [w2[i] + sl[i].start for i in range(3)]
                for i in range(3):
                    coors1[i].append( w1[i] )
                    coors2[i].append( w2[i] )
                    
            for i in range(3):
                coors1[i] = numpy.concatenate(coors1[i])
                coors2[i] = numpy.concatenate(coors2[i])
            print "coors1 = ", coors1
            print "coors2 = ", coors2
            return (coors1, coors2)
       
        #find non-zero coordinates
        fgVoxels, bgVoxels = coordinateList()
        
        self.opCarving.innerOperators[imageIndex].attachVoxelLabelsToObject(name, fgVoxels, bgVoxels)
        #mst = self.opCarving.innerOperators[imageIndex]._mst 
        #mst.object_seeds_fg_voxels[name] = fgVoxels
        #mst.object_seeds_bg_voxels[name] = bgVoxels
        
        self.opLabeling.LabelDelete.setValue(2)
        self.opLabeling.LabelDelete.setValue(1)
        self.opLabeling.LabelDelete.setValue(-1)
        # trigger a re-computation
        self.opCarving.innerOperators[imageIndex].Trigger.setDirty(slice(None))
    
    def doneObjectNamesForPosition(self, position3d, imageIndex):
        return self.opCarving.innerOperators[imageIndex].doneObjectNamesForPosition(position3d)
        
    def loadObject(self, name, imageIndex):
        print "want to load object with name = %s" % name
        fgVoxels, bgVoxels = self.opCarving.innerOperators[imageIndex].loadObject(name)
        
        #if we want to supervoxelize the seeds, do this:
        #self.opLabeling.LabelInputs[imageIndex][:] = self.opCarving.innerOperators[imageIndex]._mst.seeds[:]
        
        #else:
        shape = self.opLabeling.LabelImages[imageIndex].meta.shape
        dtype = self.opLabeling.LabelImages[imageIndex].meta.dtype
        print "shape=", shape, "dtype=", dtype
        z = numpy.zeros(shape, dtype=dtype)
        z[fgVoxels] = 2
        z[bgVoxels] = 1
        self.opLabeling.LabelInputs[imageIndex][:] = z[:]

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class OpCarving(Operator):
    name = "Carving"
    category = "interactive segmentation"
    
    # I n p u t s #
    
    #filename of the pre-processed carving graph file 
    CarvingGraphFile = InputSlot()
    
    #raw data on which carving works
    RawData      = InputSlot() 
    
    #write the seeds that the users draw into this slot 
    WriteSeeds   = InputSlot() 
    
    #trigger an update by writing into this slot
    Trigger      = InputSlot(value = numpy.zeros((1,), dtype=numpy.uint8))
   
    #number between 0.0 and 1.0 
    #bias of the background
    #FIXME: correct name?
    BackgroundPriority = InputSlot()
    
    #a number between 0 and 256
    #below the number, no background bias will be applied to the edge weights
    NoBiasBelow        = InputSlot()
    
    # O u t p u t s #
    
    Segmentation = OutputSlot()
    Supervoxels  = OutputSlot()
    DoneObjects  = OutputSlot()
    
    def __init__(self, carvingGraphFilename, *args, **kwargs):
        super(OpCarving, self).__init__(*args, **kwargs)
        print "[Carving id=%d] CONSTRUCTOR" % id(self) 
        
        #
        # FIXME: this operator has state
        #
        self._mst = MSTSegmentor.loadH5(carvingGraphFilename,  "graph")
        self._nExecutingThreads = 0
        self._cond = threading.Condition()
        
        self._currObjectName = ""
    
    def setupOutputs(self):
        self.Segmentation.meta.assignFrom(self.RawData.meta)
        self.Supervoxels.meta.assignFrom(self.RawData.meta)
        self.DoneObjects.meta.assignFrom(self.RawData.meta)
        
        self.Trigger.meta.shape = (1,)
        self.Trigger.meta.dtype = numpy.uint8
    
    def doneObjectNamesForPosition(self, position3d):
        assert len(position3d) == 3
        with self._cond:
            while self._nExecutingThreads > 0:
                self._cond.wait()
          
            #find the supervoxel that was clicked 
            sv = self._mst.regionVol[position3d]
            print "clicked on supervoxel = %r" % sv
            names = []
            for name, objectSupervoxels in self._mst.object_lut.iteritems(): 
                if numpy.sum(sv == objectSupervoxels) > 0: 
                    names.append(name)
            print "click on %r, supervoxel=%d: %r" % (position3d, sv, names)
            return names
        
    def attachVoxelLabelsToObject(self, name, fgVoxels, bgVoxels):
        with self._cond:
            while self._nExecutingThreads > 0:
                self._cond.wait()
                
            self._mst.object_seeds_fg_voxels[name] = fgVoxels
            self._mst.object_seeds_bg_voxels[name] = bgVoxels
                
    def loadObject(self, name):      
        with self._cond:
            while self._nExecutingThreads > 0:
                self._cond.wait()
                
            objNr = self._mst.object_names[name]
            print "   --> Loading object %r from nr %r" % (name, objNr)
    
            lut_segmentation = self._mst.segmentation.lut[:]
            lut_objects = self._mst.objects.lut[:]
            lut_seeds = self._mst.seeds.lut[:]
            # clean seeds
            lut_seeds[:] = 0
    
            # set foreground and background seeds
            fgVoxels = self._mst.object_seeds_fg_voxels[name]
            bgVoxels = self._mst.object_seeds_bg_voxels[name]
           
            #supervoxelize seeds: 
            #obj_seeds_fg = self._mst.object_seeds_fg[name]
            #obj_seeds_bg = self._mst.object_seeds_bg[name]
            #lut_seeds[obj_seeds_fg] = 2
            #lut_seeds[obj_seeds_bg] = 1
            
            #user-drawn seeds:
            self._mst.seeds[:] = 0
            self._mst.seeds[fgVoxels] = 2
            self._mst.seeds[bgVoxels] = 1
    
            #old way of setting current segmentation
            # set current segmentation
            #lut_segmentation[:] = numpy.where( lut_objects == objNr, 2, 1)
           
            newSegmentation = numpy.ones(len(lut_objects), dtype=numpy.int32) 
            newSegmentation[ self._mst.object_lut[name] ] = 2
            lut_segmentation[:] = newSegmentation
            del self._mst.object_lut[name]
            
            self._currObjectName = name
            
            return (fgVoxels, bgVoxels)
    
    def saveCurrentObject(self):  
        if self._currObjectName:
            print "saving object %s" % self._currObjectName
            self.saveObjectAs(self._currObjectName)
    
    def saveObjectAs(self, name): 
        with self._cond:
            while self._nExecutingThreads > 0:
                self._cond.wait()
                
            seed = 2
            print "   --> Saving object %r from seed %r" % (name, seed)
            if self._mst.object_names.has_key(name):
                objNr = self._mst.object_names[name]
            else:
                # find free objNr
                if len(self._mst.object_names.values())> 0:
                    objNr = numpy.max(numpy.array(self._mst.object_names.values())) + 1
                else:
                    objNr = 1
    
            #delete old object, if it exists
            lut_objects = self._mst.objects.lut[:]
            lut_objects[:] = numpy.where(lut_objects == objNr, 0, lut_objects)
    
            #save new object 
            lut_segmentation = self._mst.segmentation.lut[:]
            lut_objects[:] = numpy.where(lut_segmentation == seed, objNr, lut_objects)
            
            objectSupervoxels = numpy.where(lut_segmentation == seed)
            print "supervoxels = %r" % objectSupervoxels
            self._mst.object_lut[name] = objectSupervoxels
    
            #save object name with objNr
            self._mst.object_names[name] = objNr
    
            lut_seeds = self._mst.seeds.lut[:]
            print  "nonzero fg seeds shape: ",numpy.where(lut_seeds == seed)[0].shape
      
            # save object seeds
            self._mst.object_seeds_fg[name] = numpy.where(lut_seeds == seed)[0]
            self._mst.object_seeds_bg[name] = numpy.where(lut_seeds == 1)[0] #one is background=
           
            # reset seeds 
            self._mst.seeds[:] = numpy.int32(-1) #see segmentation.pyx: -1 means write zeros
            
            self._currObjectName = name
            
        #now release the lock!
    
    def execute(self, slot, roi, result):
        with self._cond:
            self._nExecutingThreads += 1
            self._cond.notify()
        
        if self._mst is None:
            return
        sl = roi.toSlice()
        if slot == self.Segmentation:
            result[0,:,:,:,0] = self._mst.segmentation[sl[1:4]]
        elif slot == self.Supervoxels:
            result[0,:,:,:,0] = self._mst.regionVol[sl[1:4]]
        elif slot  == self.DoneObjects:
            
            done_lut = numpy.zeros(len(self._mst.objects.lut), dtype=numpy.int32) 
          
            print "building done" 
            for name, objectSupervoxels in self._mst.object_lut.iteritems(): 
                print " ... name"
                done_lut[objectSupervoxels] += 1
            print "building done done, %d nonzero" % numpy.sum(done_lut > 0)
                
            result[0,:,:,:,0] = done_lut[self._mst.regionVol[sl[1:4]]]
            print "building done done, result %d nonzero" % numpy.sum(result > 0)
            
        else:
            raise RuntimeError("unknown slot")
        
        with self._cond:
            self._nExecutingThreads -= 1
            self._cond.notify()
            
        return result    
    
    def setInSlot(self, slot, key, value):
        if slot == self.WriteSeeds: 
            assert self._mst is not None
        
            #FIXME: Somehow, the labelingGui sends a value of 100 for the eraser,
            #       but the cython part of carving expects 255.
            #       Fix it here so that erasing of labels works.
            value = numpy.where(value == 100, 255, value[:])
            
            self._mst.seeds[key] = value
        else:
            raise RuntimeError("unknown slots")
        
    def notifyDirty(self, slot, key):
        if slot == self.Trigger or slot == self.BackgroundPriority or slot == self.NoBiasBelow: 
            if self._mst is None:
                return 
            if not self.BackgroundPriority.ready():
                return
            if not self.NoBiasBelow.ready():
                return
            
            bgPrio = self.BackgroundPriority.value
            noBiasBelow = self.NoBiasBelow.value
            print "compute new carving results with bg priority = %f, no bias below %d" % (bgPrio, noBiasBelow)
           
            labelCount = 2
            
            params = dict()
            params["prios"] = [1.0, bgPrio, 1.0] 
            params["uncertainty"] = "none" 
            params["noBiasBelow"] = noBiasBelow 
            
            unaries =  numpy.zeros((self._mst.numNodes,labelCount+1)).astype(numpy.float32)
            #assert numpy.sum(self._mst.seeds > 2) == 0, "seeds > 2 at %r" % numpy.where(self._mst.seeds > 2)
            self._mst.run(unaries, **params)
            
            self.Segmentation.setDirty(slice(None))
        elif slot == self.CarvingGraphFile:
            fname = self.CarvingGraphFile.value
            print "[Carving id=%d] loading graph file %s" % (id(self), fname) 
            self._mst = MSTSegmentor.loadH5(fname,  "graph")
            
            self.Segmentation.setDirty(slice(None))
        else:
            super(OpCarving, self).notifyDirty(slot, key) 
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CarvingGui(LabelingGui):
    def __init__(self, labelingSlots, observedSlots, drawerUiPath=None, rawInputSlot=None,
                 carvingApplet=None):
        # We provide our own UI file (which adds an extra control for interactive mode)
        print __file__
        directory = os.path.split(__file__)[0]
        carvingDrawerUiPath = os.path.join(directory, 'carvingDrawer.ui')

        super(CarvingGui, self).__init__(labelingSlots, observedSlots, carvingDrawerUiPath, rawInputSlot)
        self._carvingApplet = carvingApplet
        
        #set up keyboard shortcuts
        c = QShortcut(QKeySequence("Ctrl+S"), self, member=self.labelingDrawerUi.segment.click, ambiguousMember=self.labelingDrawerUi.segment.click)

        def onSegmentButton():
            print "segment button clicked"
            self._carvingApplet.topLevelOperator.opCarving.Trigger[0].setDirty(slice(None))
        self.labelingDrawerUi.segment.clicked.connect(onSegmentButton)
        
        def onBackgroundPrioritySpin(value):
            print "background priority changed to %f" % value
            self._carvingApplet.topLevelOperator.opCarving.BackgroundPriority.setValue(value)
        self.labelingDrawerUi.backgroundPrioritySpin.valueChanged.connect(onBackgroundPrioritySpin)
        
        def onNoBiasBelowSpin(value):
            print "background priority changed to %f" % value
            self._carvingApplet.topLevelOperator.opCarving.NoBiasBelow.setValue(value)
        self.labelingDrawerUi.noBiasBelowSpin.valueChanged.connect(onNoBiasBelowSpin)
        
        def onSaveAsButton():
            print "save object as?"
            from PyQt4.QtGui import QInputDialog
            name, ok = QInputDialog.getText(self, 'Save Object As', 'object name') 
            name = str(name)
            print "save object as %s" % name
            if ok:
                self._carvingApplet.topLevelOperator.saveObjectAs(name, self.imageIndex)
        self.labelingDrawerUi.saveAs.clicked.connect(onSaveAsButton)
        
        def onSaveButton():
            self._carvingApplet.topLevelOperator.saveCurrentObject(self.imageIndex)
        self.labelingDrawerUi.save.clicked.connect(onSaveButton)
        
        def onLoadObjectButton():
            print "load which object?"
            from PyQt4.QtGui import QInputDialog
            name, ok = QInputDialog.getText(self, 'Load Object', 'object name') 
            name = str(name)
            print "load object %s" % name
            if ok:
                self._carvingApplet.topLevelOperator.loadObject(name, self.imageIndex)
        self.labelingDrawerUi.load.clicked.connect(onLoadObjectButton)
        
        def labelBackground():
            self.selectLabel(0)
        def labelObject():
            self.selectLabel(1)
       
        self._labelControlUi.labelListModel.allowRemove(False) 
        
        QShortcut(QKeySequence("1"), self, member=labelBackground, ambiguousMember=labelBackground)
        QShortcut(QKeySequence("2"), self, member=labelObject, ambiguousMember=labelObject)
       
        def layerIndexForName(name): 
            return self.layerstack.findMatchingIndex(lambda x: x.name == name)
        
        def toggleDone():
            row = layerIndexForName("done")
            self.layerstack.selectRow(row)
            layer = self.layerstack[row]
            layer.visible = not layer.visible
            self.viewerControlWidget().layerWidget.setFocus()
        QShortcut(QKeySequence("d"), self, member=toggleDone, ambiguousMember=toggleDone)
        
        def toggleSegmentation():
            row = layerIndexForName("segmentation")
            self.layerstack.selectRow(row)
            layer = self.layerstack[row]
            layer.visible = not layer.visible
            self.viewerControlWidget().layerWidget.setFocus()
        QShortcut(QKeySequence("s"), self, member=toggleSegmentation, ambiguousMember=toggleSegmentation)
    
    def handleEditorRightClick(self, currentImageIndex, position5d, globalWindowCoordinate):
        print "handleEd"
        names = self._carvingApplet.topLevelOperator.doneObjectNamesForPosition(position5d[1:4], currentImageIndex)
       
        m = QMenu(self)
        m.addAction("position %d %d %d" % (position5d[1], position5d[2], position5d[3]))
        for n in names:
            m.addAction("edit %s" % n)
            m.addAction("delete %s" % n)
            
        act = m.exec_(globalWindowCoordinate) 
        for n in names:
            if act.text() == "edit %s" %n:
                self._carvingApplet.topLevelOperator.loadObject(n, self.imageIndex)
        
    def getNextLabelName(self):
        l = len(self._labelControlUi.labelListModel)
        if l == 0:
            return "Background"
        else:
            return "Object"
        
    def appletDrawers(self):
        return [ ("Carving", self._labelControlUi) ]

    def setupLayers( self, currentImageIndex ):
        layers = []
        
        # Labels
        labellayer, labelsrc = self.createLabelLayer(currentImageIndex)
        if labellayer is not None:
            layers.append(labellayer)
            # Tell the editor where to draw label data
            self.editor.setLabelSink(labelsrc)
       
        #segmentation 
        #seg = self._carvingApplet._segmentation5D[currentImageIndex]
        seg = self._carvingApplet.topLevelOperator.opCarving.Segmentation[currentImageIndex]
        if seg.ready(): 
            #source = RelabelingArraySource(seg)
            #source.setRelabeling(numpy.arange(256, dtype=numpy.uint8))
            colortable = [QColor(0,0,0,0).rgba(), QColor(0,0,0,0).rgba(), QColor(0,255,0).rgba()]
            for i in range(256-len(colortable)):
                r,g,b = numpy.random.randint(0,255), numpy.random.randint(0,255), numpy.random.randint(0,255)
                colortable.append(QColor(r,g,b).rgba())
            layer = ColortableLayer(LazyflowSource(seg), colortable)
            #layer = ColortableLayer(source, colortable)
            layer.name = "segmentation"
            layer.visible = True
            layer.opacity = 0.3
            layers.append(layer)
        
        #segmentation 
        done = self._carvingApplet.topLevelOperator.opCarving.DoneObjects[currentImageIndex]
        if done.ready(): 
            colortable = [QColor(0,0,0,0).rgba(), QColor(0,0,255).rgba()]
            for i in range(254-len(colortable)):
                r,g,b = numpy.random.randint(0,255), numpy.random.randint(0,255), numpy.random.randint(0,255)
                colortable.append(QColor(r,g,b).rgba())
            layer = ColortableLayer(LazyflowSource(done), colortable)
            #layer = ColortableLayer(source, colortable)
            layer.name = "done"
            layer.visible = False
            layer.opacity = 0.5
            layers.append(layer)
            
        #supervoxel
        sv = self._carvingApplet.topLevelOperator.opCarving.Supervoxels[currentImageIndex]
        if sv.ready():
            for i in range(256):
                r,g,b = numpy.random.randint(0,255), numpy.random.randint(0,255), numpy.random.randint(0,255)
                colortable.append(QColor(r,g,b).rgba())
            layer = ColortableLayer(LazyflowSource(sv), colortable)
            #layer = ColortableLayer(source, colortable)
            layer.name = "supervoxels"
            layer.visible = False
            layer.opacity = 1.0
            layers.append(layer)
        
        #raw data
        img = self._carvingApplet._inputImage[currentImageIndex]
        if img.ready():
            layer = self.createStandardLayerFromSlot( img )
            layer.name = "raw"
            layer.visible = True
            layer.opacity = 1.0
            #layers.insert(1, layer)
            layers.append(layer)
        
        return layers

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CarvingApplet(LabelingApplet):
    def __init__(self, graph, projectFileGroupName, raw, carvingGraphFile):
        super(CarvingApplet, self).__init__(graph, projectFileGroupName)

        self._raw = raw
       
        #
        # raw data
        # 
        o = OperatorWrapper( adaptors.Op5ifyer(graph) )
        o.order.setValue('txyzc')
        o.input.connect(self._raw)
        self._inputImage = o.output

        self._topLevelOperator = OpCarvingTopLevel( carvingGraphFile, graph=graph )
        self._topLevelOperator.opCarving.RawData.connect(self._inputImage)
        self._topLevelOperator.opCarving.BackgroundPriority.setValue(0.95)
        self._topLevelOperator.opCarving.NoBiasBelow.setValue(64)
        #self.opCarving.CarvingGraphFile.setValue(carvingGraphFile)

        o = OperatorWrapper( adaptors.Op5ifyer(graph) )
        o.order.setValue('txyzc')
        o.input.connect(self._topLevelOperator.opCarving.Segmentation)
        self._segmentation5D = o.output
        assert self._segmentation5D is not None

    @property
    def gui(self):
        if self._gui is None:

            labelingSlots = LabelingGui.LabelingSlots()
            labelingSlots.labelInput = self.topLevelOperator.opLabeling.LabelInputs
            labelingSlots.labelOutput = self.topLevelOperator.opLabeling.LabelImages
            labelingSlots.labelEraserValue = self.topLevelOperator.opLabeling.LabelEraserValue
            labelingSlots.labelDelete = self.topLevelOperator.opLabeling.LabelDelete
            labelingSlots.maxLabelValue = self.topLevelOperator.opLabeling.MaxLabelValue
            labelingSlots.labelsAllowed = self.topLevelOperator.opLabeling.LabelsAllowedFlags
            
            self._gui = CarvingGui( labelingSlots, [self._segmentation5D, self._inputImage], rawInputSlot=self._raw, carvingApplet=self )
        return self._gui

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class CarvingWorkflow(Workflow):
    
    def __init__(self, carvingGraphFile):
        super(CarvingWorkflow, self).__init__()
        self._applets = []

        graph = Graph()
        self._graph = graph

        ## Create applets 
        self.projectMetadataApplet = ProjectMetadataApplet()
        self.dataSelectionApplet = DataSelectionApplet(graph, "Input Data", "Input Data", supportIlastik05Import=True, batchDataGui=False)
        self.viewerApplet = LayerViewerApplet(graph)

        self.carvingApplet = CarvingApplet(graph, "xxx", self.dataSelectionApplet.topLevelOperator.Image, carvingGraphFile)
        self.carvingApplet.topLevelOperator.RawData.connect( self.dataSelectionApplet.topLevelOperator.Image )
        self.carvingApplet.topLevelOperator.opLabeling.LabelsAllowedFlags.connect( self.dataSelectionApplet.topLevelOperator.AllowLabels )
        self.carvingApplet.gui.minLabelNumber = 2
        self.carvingApplet.gui.maxLabelNumber = 2

        ## Access applet operators
        opData = self.dataSelectionApplet.topLevelOperator
        
        ## Connect operators ##
        
        self._applets.append(self.projectMetadataApplet)
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.carvingApplet)

        # The shell needs a slot from which he can read the list of image names to switch between.
        # Use an OpAttributeSelector to create a slot containing just the filename from the OpDataSelection's DatasetInfo slot.
        opSelectFilename = OperatorWrapper( OpAttributeSelector(graph=graph) )
        opSelectFilename.InputObject.connect( opData.Dataset )
        opSelectFilename.AttributeName.setValue( 'filePath' )

        self._imageNameListSlot = opSelectFilename.Result

    def setCarvingGraphFile(self, fname):
        self.carvingApplet.topLevelOperator.opCarving.CarvingGraphFile.setValue(fname)

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self._imageNameListSlot
    
    @property
    def graph( self ):
        '''the lazyflow graph shared by the applets'''
        return self._graph

if __name__ == "__main__":
    import lazyflow
    import numpy
    from ilastik.shell.gui.startShellGui import startShellGui
    import socket
    
    graph = lazyflow.graph.Graph()
    
    from optparse import OptionParser
    usage = "%prog [options] <carving graph filename> <project filename to be created>"
    parser = OptionParser(usage)

#    import sys
#    sys.argv.append("/Users/bergs/Documents/workspace/applet-workflows/denk.h5")
#    sys.argv.append("test.ilp")

    (options, args) = parser.parse_args()
    
    if len(args) == 2:
        carvingGraphFilename = args[0]
        projectFilename = args[1]
        def loadProject(shell, workflow):
            shell.createAndLoadNewProject(projectFilename)
            workflow.setCarvingGraphFile(carvingGraphFilename)
            # Add a file
            from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
            info = DatasetInfo()
            info.filePath = carvingGraphFilename + "/graph/raw"
            opDataSelection = workflow.dataSelectionApplet.topLevelOperator
            opDataSelection.Dataset.resize(1)
            opDataSelection.Dataset[0].setValue(info)
            shell.setSelectedAppletDrawer(2)
        startShellGui( CarvingWorkflow, loadProject, windowTitle="Carving %s" % carvingGraphFilename,
                       workflowKwargs={'carvingGraphFile': carvingGraphFilename} )
    else:
        parser.error("incorrect number of arguments")
