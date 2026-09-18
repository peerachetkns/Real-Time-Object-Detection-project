import numpy as np
import cv2 
# ไม่จำเป็นต้องใช้ helper แล้ว เนื่องจากเราฟิกซ์สีตามโจทย์ (กรอบแดง, รถเขียว)

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

cap = cv2.VideoCapture("road2.mp4")

# Load the Caffe model 
# net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel") 
net = cv2.dnn.readNetFromCaffe("vgg_ssd.prototxt", "vgg_ssd.caffemodel")

# หมายเหตุ: หากรันบน Mac Apple Silicon (M1/M2/M3) ให้คอมเมนต์ 2 บรรทัดนี้ทิ้ง เพราะไม่รองรับ CUDA
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break # จบวิดีโอ

    height = frame.shape[0]  
    width = frame.shape[1] 

    # ---------------------------------------------------------
    # 1. กำหนดพิกัด ROI (Region of Interest) สำหรับเลนกลาง
    # (สามารถปรับค่า % ตัวคูณด้านล่างให้พอดีกับวิดีโอจริงได้)
    # ---------------------------------------------------------
    roi_x1 = int(width * 0.30)  # ขอบซ้าย (ประมาณ 30% ของความกว้าง)
    roi_y1 = int(height * 0.55) # ขอบบน (ประมาณ 55% ของความสูง)
    roi_x2 = int(width * 0.70)  # ขอบขวา (ประมาณ 70% ของความกว้าง)
    roi_y2 = int(height * 0.95) # ขอบล่าง (ประมาณ 95% ของความสูง)

    # วาดกรอบ ROI สีแดงลงบนภาพ (BGR: 0, 0, 255)
    
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

    # แปลงภาพเข้า Network
    #MobileNetSSD
    # blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False) 
    #vgg_ssd
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 117, 123), False)
    net.setInput(blob)
    detections = net.forward()

    # วนลูปตรวจสอบผลลัพธ์
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] 
        
        if confidence > 0.3: 
            class_id = int(detections[0, 0, i, 1]) 
            
            # ---------------------------------------------------------
            # 2. กรองเฉพาะคลาส 'car' (class_id == 7) เท่านั้น
            # ---------------------------------------------------------
            if class_id == 7: 
                # สเกลพิกัดกรอบของรถที่ตรวจจับได้
                xLeftBottom = int(width * detections[0, 0, i, 3]) 
                yLeftBottom = int(height * detections[0, 0, i, 4])
                xRightTop   = int(width * detections[0, 0, i, 5])
                yRightTop   = int(height * detections[0, 0, i, 6])

                # หาจุดศูนย์กลางของรถคันนี้
                car_center_x = (xLeftBottom + xRightTop) // 2
                car_center_y = (yLeftBottom + yRightTop) // 2

                # ---------------------------------------------------------
                # 3. ตรวจสอบว่าจุดศูนย์กลางของรถ อยู่ในกรอบ ROI สีแดงหรือไม่
                # ---------------------------------------------------------
                if (roi_x1 < car_center_x < roi_x2) and (roi_y1 < car_center_y < roi_y2):
                    
                    # ถ้าอยู่ใน ROI ให้วาดกรอบสีเขียว (BGR: 0, 255, 0)
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0), 2)

                    # วาดป้ายชื่อ Label (เอาแค่คำว่า car ไม่ต้องโชว์เลข confidence ตามภาพโจทย์)
                    label = classNames[class_id]
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    y_label = max(yLeftBottom, labelSize[1])
                    # วาดพื้นหลังป้ายชื่อสีเขียว
                    cv2.rectangle(frame, (xLeftBottom, y_label - labelSize[1]),
                                            (xLeftBottom + labelSize[0], y_label + baseLine),
                                            (0, 255, 0), cv2.FILLED)
                    # ใส่ตัวหนังสือสีดำ
                    cv2.putText(frame, label, (xLeftBottom, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Break with ESC 
        break

cap.release()
cv2.destroyAllWindows()
