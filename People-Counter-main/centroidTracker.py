from  scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self,maxDisappeared=5, maxDistance=50):
        self.nextObjectId=0
        self.objects=OrderedDict()
        self.disappeared=OrderedDict()
        self.maxDisappeared=maxDisappeared
        self.maxDistance = maxDistance

    def register(self,centeroid):
        self.objects[self.nextObjectId]=centeroid
        self.disappeared[self.nextObjectId]=0
        self.nextObjectId+=1

    def deregister(self,objectId):
        del self.objects[objectId]
        del self.disappeared[objectId]

    def update(self,rects):
        if len(rects)==0:
            for objectId in list(self.disappeared.keys()):
                self.disappeared[objectId]+=1
                if self.disappeared[objectId]>self.maxDisappeared:
                    self.deregister(objectId)
            return self.objects
        
        inputCentroids=np.zeros((len(rects),2),dtype="int")
        for (i,(startX,startY,endX,endY))in enumerate(rects):
            cX=int((startX+endX)/2.0)
            cY=int((startY+endY)/2.0)
            inputCentroids[i]=(cX,cY)
        
        if len(self.objects)==0:
            for i in range(0,len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIds=list(self.objects.keys())
            objectCentroids=list(self.objects.values())

            D=dist.cdist(np.array(objectCentroids),inputCentroids)
            rows=D.min(axis=1).argsort()
            cols=D.argmin(axis=1)[rows]
            usedRows=set()
            usedCols=set()
            for(row,col)in zip(rows,cols):
                if(row in usedRows or col in usedCols):
                    continue
                if D[row, col] > self.maxDistance:continue

                objectId=objectIds[row]
                self.objects[objectId]=inputCentroids[col]
                self.disappeared[objectId]=0
                usedRows.add(row)
                usedCols.add(col)
                unusedRows=set(range(0,D.shape[0])).difference(usedRows)
                unusedCols=set(range(0,D.shape[0])).difference(usedCols)
                if D.shape[0]>=D.shape[1]:
                    for rows in unusedRows:
                        objectId=objectIds[row]
                        self.disappeared[objectId]+=1
                        if self.disappeared[objectId]>self.maxDisappeared:
                            self.deregister[objectId]
                else:
                    for col in unusedCols:
                        self.register(inputCentroids[col])
        return self.objects