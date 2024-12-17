import cv2
from ultralytics import YOLO
import numpy as np
import torch

def open_camera():
    capture = cv2.VideoCapture(0)
    return capture

def object_tracking(capture):
    # load a pre trained model
    model = YOLO("yolo11n.pt")
    print(torch.backends.mps.is_available())

    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    if not ret:
        print("unable to open")
        return exit(1)

    #32 sports ball and 67 for cell phone classes
    results = model.track(frame, agnostic_nms = True, stream = True, persist = True, classes = [67], tracker="bytetrack.yaml")

    annotated_frame = frame.copy()
    center = (None,None)
    center2 = (None, None)

    object_detection = []

    for result in results:
        annotated_frame = result.plot()

        for box in result.boxes:
            class_index = int(box.cls)
            class_name = result.names[class_index]
            track_id = box.id
            object_detection.append((box, class_index, class_name, track_id))
            if track_id is None:
                continue
            for box, class_index, class_name, track_id in object_detection:
                track_id = int(track_id.item())
                x, y, w, h = box.xywh[0].cpu().numpy()
                if track_id == 1:
                    center = (int(x + w / 2), int(y + h / 2))
                elif track_id == 2:
                    center2 = (int(x + w / 2), int(y + h / 2))

        cv2.imshow("YOLO Inference", annotated_frame)

    return frame, center, center2



