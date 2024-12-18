import cv2
from ultralytics import YOLO
import numpy as np
import torch

def open_camera():
    capture = cv2.VideoCapture(0)
    return capture

def object_tracking(capture):
    # load a pre trained model
    model = YOLO("yolo11s.pt")
    print(torch.backends.mps.is_available())

    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    if not ret:
        print("unable to open")
        return exit(1)

    #32 sports ball and 67 for cell phone classes
    results = model.track(frame, agnostic_nms = True, stream = True, persist = True, classes = [32,67], tracker="bytetrack.yaml")

    #copies og frame
    annotated_frame = frame.copy()
    #initializes center to none in case there's no object that matches
    center = (None,None)
    center2 = (None, None)

    #array that stores objects detected
    object_detection = []

    for result in results:
        annotated_frame = result.plot()

        for box in result.boxes:
            class_index = int(box.cls)
            class_name = result.names[class_index]
            track_id = box.id
            #add object detected to array
            object_detection.append((box, class_index, class_name, track_id))
            if track_id is None:
                continue
            for box, class_index, class_name, track_id in object_detection:
                track_id = int(track_id.item())
                #calculates coordinates of bounding box
                x, y, w, h = box.xywh[0].cpu().numpy()
                #tracks cellphone for one paddle
                if class_name == 'cell phone':
                    center = (int(x + w / 2), int(y + h / 2))
                #tracks sports ball for the other paddle
                elif class_name == 'sports ball':
                    center2 = (int(x + w / 2), int(y + h / 2))

        cv2.imshow("Annotated Frame", annotated_frame)

    return frame, center, center2


