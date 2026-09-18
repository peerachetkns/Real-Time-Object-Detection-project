import numpy as np
import cv2 
import helper

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

classColors = helper.getRandomColors(20)
print(classColors)

cap = cv2.VideoCapture("road3.mp4")

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel") 
# net = cv2.dnn.readNetFromCaffe("vgg_ssd.prototxt", "vgg_ssd.caffemodel")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # MobileNet requires fixed dimensions for input image(s)
    
    #MobileNetSSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False) 

    #vgg_ssd
    # blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 117, 123), False)

    #Set to network the input blob 
    net.setInput(blob)

    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    # Size of frame
    height = frame.shape[0]  
    width = frame.shape[1] 

    # Locate location and class of object detected
    # There is a fix index for class, location and confidence
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > 0.3: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label
            
            # Scale detection frame
            xLeftBottom = int(width * detections[0, 0, i, 3]) 
            yLeftBottom = int(height * detections[0, 0, i, 4])
            xRightTop   = int(width * detections[0, 0, i, 5])
            yRightTop   = int(height * detections[0, 0, i, 6])

            # Draw location
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),classColors[class_id])

            # Draw label and confidence
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
