import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time



def detect_pre(frame, facenet, maskNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224), (104.0, 177.0, 123.0))
    facenet.setInput(blob)
    detections = facenet.forward()
    print(detections.shape)
    faces=[]
    locs=[]
    preds=[]
    for i in range(0, detections.shape[2]):
        conf = detections[0,0,i,2];
        print(conf)
        if conf>0.5:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy) = box.astype("int")
            (startx,starty) = (max(0, startx), max(0,starty))
            (endx,endy) = (min(w-1,endx),max(h-1,endy))
            face = frame[starty:endy, startx:endx]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face =img_to_array(face)
            face =preprocess_input(face)
            faces.append(face)
            locs.append((startx, starty, endx, endy))
            
            
    if len(faces)>0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs,preds)
    

prototxt = r"face_detector\deploy.prototxt"
caffe = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
facenet = cv2.dnn.readNet(prototxt, caffe)
maskNet = load_model("mask_detector.model")
print("Video stream")
vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    (loc,pred) = detect_pre(frame, facenet, maskNet)
    
    for (box, pred) in zip(loc,pred):
        (startx,starty,endx,endy) = box
        (mask, withoutMask) = pred
        
        print(mask,withoutMask)
        label = "Mask" if mask>withoutMask else "No mask"
        color = (0,255,0) if label=='Mask' else (0,0,255)
        
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startx, starty-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.rectangle(frame, (startx,starty), (endx,endy), color, 2)
        
    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break
        
cv2.destroyAllWindows()
vs.stop()
        
        
        