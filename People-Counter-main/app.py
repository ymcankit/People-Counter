from typing import Counter
from flask import Flask, render_template, Response, request, redirect, url_for,jsonify
import cv2
import os
from centroidTracker import CentroidTracker
import numpy as np
import dlib
import time
from trackableobject import TrackableObject
from werkzeug.utils import secure_filename

upload_folder = 'uploads'

app = Flask(__name__)
app.config['upload_folder'] = upload_folder
address = []
countRl = [1]
modelFIle = "MobileNetSSD/MobileNetSSD_deploy.caffemodel"
protext = "MobileNetSSD/MobileNetSSD_deploy.prototxt.txt"

classNames = {0: 'background',
              1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
              10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
              14: 'motorbike', 15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}


classes = None
net = cv2.dnn.readNet('yolo/yolov3.weights', 'yolo/yolov3.cfg')


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

with open('label.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]


class check():
    def __init__(self):
        self.is_decoded = False
        self.count=0
        self.numberOfLines=0
        self.startRef=[]
        self.endRef=[]
        self.orientation=0
        self.Inp=0
        self.Outp=0

def get_check():
    return check()

def getFrameYolo(obj):
    cap = cv2.VideoCapture(address[0])
    ct=CentroidTracker(maxDisappeared=10 ,maxDistance=90)
    trackers = []
    rects=[]
    objects_id=[]
    trackableObjects = {}
    count=0
    LE=100
    RE=0
    totalIn=0
    totalDown=0
    count=0
    W=None
    H=None
    while(True):
        ret, frame = cap.read()
        if (ret == False):
            obj.is_decoded=True
            cap.release()
            cv2.destroyAllWindows()
            break
        frame = cv2.resize(frame, (500, 500), fx=0, fy=0,
                        interpolation=cv2.INTER_CUBIC)
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            RE=W-180
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        rects=[]
        if (count % 4 == 0 or count % 5 == 0 or count % 6==0):
            trackers=[]
            net.setInput(cv2.dnn.blobFromImage(frame, 0.00392,
                        (416, 416), (0, 0, 0), True, crop=False))
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            Width = frame.shape[1]
            Height = frame.shape[0]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
                        
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
            # check if is people detection
            for i in indices:
                i = i[0]
                box = boxes[i]
                if class_ids[i] == 0:
                    label = str(classes[class_id])
                    cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(
                        box[0]+box[2]), round(box[1]+box[3])), (0, 0, 255), 2)
                    # rects.append((round(box[0]), round(box[1]), round(
                    #     box[0]+box[2]), round(box[1]+box[3])))
                    cv2.putText(frame, label, (round(
                        box[0])-10, round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    tracker=dlib.correlation_tracker()
                    rt=dlib.rectangle(round(box[0]), round(box[1]), round(box[0]+box[2]), round(box[1]+box[3]))
                    tracker.start_track(rgb,rt)
                    trackers.append(tracker)
        else :
            for tracker in trackers:
                    tracker.update(rgb)
                    cod=tracker.get_position()
                    startX=int(cod.left())
                    startY=int(cod.top())
                    endX=int(cod.right())
                    endY=int(cod.bottom())
                    cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                    rects.append((startX,startY,endX,endY))
        objects=ct.update(rects)
        
        
        for(objectId ,centroid)in objects.items():
            print("objectId-->",objectId)
            to = trackableObjects.get(objectId, None)
            if to is None:
                to = TrackableObject(objectId, centroid)
            else:
                    if obj.orientation==2:
                        y = [c[0] for c in to.centroids]
                        direction = centroid[0] - np.mean(y)
                        to.centroids.append(centroid)
                        if not to.counted:
                            (startRecX,startRecY)=obj.startRef[0]
                            if direction<0 and centroid[0] <int(startRecX):
                                print("woprking")
                                totalIn = totalIn+1
                                obj.Inp=totalIn
                                obj.count=obj.count+1
                                to.counted = True
                            if check1.numberOfLines==2:
                                (startRecX,startRecY)=obj.startRef[1]
                                if direction>0 and centroid[0] >int(startRecX):
                                    print("woprking")
                                    totalIn = totalIn+1
                                    obj.Inp=totalIn
                                    obj.count=obj.count+1
                                    to.counted = True
                    else:
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)
                        if not to.counted:
                            (startRecX,startRecY)=obj.startRef[0]
                            if direction<0 and centroid[1] <int(startRecY):
                                print("woprking")
                                totalDown = totalDown+1
                                obj.Outp=totalDown
                                obj.count=obj.count-1
                                to.counted = True
                            if direction>0 and centroid[1] >int(startRecY):
                                    print("woprking")
                                    totalIn = totalIn+1
                                    obj.Inp=totalIn
                                    obj.count=obj.count+1
                                    to.counted = True
                            if check1.numberOfLines==2:
                                (startRecX,startRecY)=obj.startRef[1]
                                if direction<0 and centroid[1] <int(startRecY):
                                    print("woprking")
                                    totalIn = totalIn+1
                                    obj.Inp=totalIn
                                    obj.count=obj.count+1
                                    to.counted = True
                                if direction>0 and centroid[1] >int(startRecY):
                                    print("woprking")
                                    totalDown = totalDown+1
                                    obj.Outp=totalDown
                                    obj.count=obj.count-1
                                    to.counted = True
            trackableObjects[objectId]=to
            # if(centroid[0]>=RE):
            #     if objectId  not in objects_id:
            #         totalIn=totalIn+1
            #         obj.count=totalIn
            #         objects_id.append(objectId)
            # if(centroid[0]<=LE):
            #     if objectId  not in objects_id:
            #         totalIn=totalIn+1
            #         obj.count=totalIn
            #         objects_id.append(objectId)
            text = "ID {}".format(objectId)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame,(centroid[0],centroid[1]),4,(0,255,0),-1)
        intxt="IN {}".format(totalIn)
        outtxt = "OUT {}".format(totalDown)
        cv2.putText(frame,intxt,(10,50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, outtxt, (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        for i in range(check1.numberOfLines):
                (startRecX,startRecY)=check1.startRef[i]
                (endRecX,endRecY)=check1.endRef[i]
                cv2.line(frame, (startRecX, startRecY), (endRecX, endRecY), (0, 255, 255), 2)
        count = count+1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def getFrame(obj):
    cap = cv2.VideoCapture(address[0])
    model = cv2.dnn.readNetFromCaffe(protext, modelFIle)
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    rects = []
    trackers = []
    objects_id = []
    trackableObjects = {}
    count = 0
    LE = 100
    RE = 0
    totalIn = 0
    totalDown=0
    count = 0
    W = None
    H = None
    while(True):
        ret, frame = cap.read()
        if(ret == False):
            obj.is_decoded=True
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            frame = cv2.resize(frame, (500, 500), fx=0, fy=0,
                               interpolation=cv2.INTER_CUBIC)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                RE = W-180

            rects = []
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if(count % 30 == 0):
                trackers = []
                blob = cv2.dnn.blobFromImage(
                    frame, 0.007843, (300, 300), 127.5)
                model.setInput(blob)
                detection = model.forward()
                cols = frame.shape[1]
                rows = frame.shape[0]

                for i in np.arange(0, detection.shape[2]):
                    confidence = detection[0, 0, i, 2]

                    if(confidence > 0.50):
                        idx = int(detection[0, 0, i, 1])
                        if classNames[idx] != "person":
                            continue
                        box = detection[0, 0, i, 3:7] * \
                            np.array([cols, rows, cols, rows])
                        (startX, startY, endX, endY) = box.astype("int")
                        tracker = dlib.correlation_tracker()
                        rt = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rt)
                        trackers.append(tracker)
                        cv2.rectangle(frame, (startX, startY),
                                      (endX, endY), (0, 0, 255), 2)
                        label = classNames[idx]+":"+str(confidence)
                        labelSize, baseLine = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        startY = max(startY, labelSize[1])
                        cv2.rectangle(
                            frame, (startX, startY-labelSize[1]), (startX+labelSize[0], startY+baseLine), (0, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (startX, startY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            else:
                for tracker in trackers:
                    tracker.update(rgb)
                    cod = tracker.get_position()
                    startX = int(cod.left())
                    startY = int(cod.top())
                    endX = int(cod.right())
                    endY = int(cod.bottom())
                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), (0, 255, 0), 2)
                    rects.append((startX, startY, endX, endY))
            objects = ct.update(rects)

            for(objectId, centroid) in objects.items():
                to = trackableObjects.get(objectId, None)
                if to is None:
                    to = TrackableObject(objectId, centroid)
                else:
                    print("Elsee---")
                    if obj.orientation == 2:
                        y = [c[0] for c in to.centroids]
                        direction = centroid[0] - np.mean(y)
                        to.centroids.append(centroid)
                        print("inside the thing--")
                        if not to.counted:
                            (startRecX,startRecY)=obj.startRef[0]
                            if direction<0 and centroid[0] <int(startRecX):
                                print("woprking 1")
                                totalIn = totalIn+1
                                obj.Inp=totalIn
                                obj.count=obj.count+1
                                to.counted = True
                            if obj.numberOfLines==2:
                                (startRecX,startRecY)=obj.startRef[1]
                                if direction>0 and centroid[0] >int(startRecX):
                                    print("woprking 2")
                                    totalIn = totalIn+1
                                    obj.Inp=totalIn
                                    obj.count=obj.count+1
                                    to.counted = True
                    elif obj.orientation == 1:
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)
                        if not to.counted:
                            (startRecX,startRecY)=obj.startRef[0]
                            if direction<0 and centroid[1] <int(startRecY):
                                print("woprking")
                                totalDown = totalDown+1
                                obj.Outp=totalDown
                                obj.count=obj.count-1
                                to.counted = True
                            if direction>0 and centroid[1] >int(startRecY):
                                    print("woprking")
                                    totalIn = totalIn+1
                                    obj.Inp=totalIn
                                    obj.count=obj.count+1
                                    to.counted = True
                            if obj.numberOfLines==2:
                                (startRecX,startRecY)=obj.startRef[1]
                                if direction<0 and centroid[1] <int(startRecY):
                                    print("woprking")
                                    totalDown = totalDown+1
                                    obj.Outp=totalDown
                                    obj.count=obj.count-1
                                    to.counted = True
                                if direction>0 and centroid[1] >int(startRecY):
                                    print("woprking")
                                    totalIn = totalIn+1
                                    obj.Inp=totalIn
                                    obj.count=obj.count+1
                                    to.counted = True
                trackableObjects[objectId]=to
                # if(centroid[0] >= RE):
                #     if objectId not in objects_id:
                #         totalIn = totalIn+1
                #         obj.count=totalIn
                #         countRl.clear()
                #         countRl.append(totalIn)
                #         print(countRl)
                #         objects_id.append(objectId)
                # if(centroid[0] <= LE):
                #     if objectId not in objects_id:
                #         totalIn = totalIn+1
                #         obj.count=totalIn
                #         countRl.clear()
                #         countRl.append(totalIn)
                #         print(countRl)
                #         objects_id.append(objectId)

                text = "ID {}".format(objectId)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(
                    frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            intxt = "IN {}".format(totalIn)
            outtxt = "OUT {}".format(totalDown)
            cv2.putText(frame, intxt, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, outtxt, (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for i in range(check1.numberOfLines):
                (startRecX,startRecY)=check1.startRef[i]
                (endRecX,endRecY)=check1.endRef[i]
                cv2.line(frame, (int(startRecX), int(startRecY)), (endRecX, endRecY), (0, 255, 255), 2)
                

            count = count+1
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

check1=get_check()

@app.route('/video_feed')
def video_feed():
    return Response(getFrame(check1), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    return Response(getFrameYolo(check1), mimetype='multipart/x-mixed-replace;boundary=frame')
@app.route('/detect')
def main():
    return render_template('index.html')

@app.route('/detect-Yolo')
def yolo():
    return render_template('yolo-detection.html')

@app.route('/choose_detection')
def function():
    return render_template('index1.html')

@app.route('/is_decoded')
def is_decoded():
    return jsonify({'is_decoded':check1.is_decoded})

@app.route('/location', methods=['POST'])
def selectingFunction():
    if request.method == "POST":
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['upload_folder'], filename))
        address.clear()
        address.append(os.path.join(app.config['upload_folder'], filename))
        print(address[0])
        return redirect('config')
@app.route('/config')
def config():
    return render_template('/config.html')
@app.route('/config_rec',methods=['POST'])
def config_recieve():
    n=request.form.get('Selections')
    check1.orientation=int(request.form.get('orientation'))
    print("orientaion--->",check1.orientation)
    check1.startRef=[]
    check1.endRef=[]
    check1.Inp=0
    check1.Outp=0
    check1.count=0
    check1.numberOfLines=int(n)
    for i in range(int(n)):
        firstStart=int(request.form.get('start'+str(i)))
        secondStart=int(request.form.get('starta'+str(i)))
        firstEnd=int(request.form.get('end'+str(i)))
        secondEnd=int(request.form.get('enda'+str(i)))
        check1.startRef.append([firstStart,secondStart])
        check1.endRef.append([firstEnd,secondEnd])
    
    print(check1.startRef,check1.endRef)
    return redirect('choose_detection')
@app.route('/result')
def result():
    check1.is_decoded=False
    return render_template('result.html',counted=check1.count,In=check1.Inp,Out=check1.Outp)
@app.route('/')
def index():
    return render_template('file_choose.html')


if __name__ == "__main__":
    app.run(debug=True)
