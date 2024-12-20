import cv2
from ultralytics import YOLO
import logging

# Removes YOLO messages to the console
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def open_camera():
    capture = cv2.VideoCapture(0)
    return capture


def map_object_to_screen(y_obj, y_min_orig, y_max_orig, y_screen_min, y_screen_max, h_obj, h_bar):

    # Mapping of the center of the object on the space of the screen
    y_screen = y_screen_min + (y_obj - y_min_orig) / (y_max_orig - y_min_orig) * (y_screen_max - y_screen_min)

    # Calculating the offset proportional to the height difference
    offset = max(0, (h_obj - h_bar) / 2)  # Make sure offset is not negative
    proportional_factor = (y_obj - y_min_orig) / (y_max_orig - y_min_orig)

    # Adjust offset based on object position in the in interval
    # When the object is at the topo, offset is reduced
    # When object is at the bottom, offset increases
    D = offset * (2 * proportional_factor - 1)

    # Adjust position on the screen according to offset
    y_screen_adjusted = y_screen + D

    # Make sure adjusted position is between the screen limits
    y_screen_adjusted = max(y_screen_min, min(y_screen_adjusted, y_screen_max))

    return y_screen_adjusted


def process_frame_half(frame_half, model, height, screen_height):
    center = (None, None)

    results = model.track(
        frame_half,
        agnostic_nms=True,
        persist=True,
        classes=[67],  # Class 67 = "cell phone"
        tracker="bytetrack.yaml"
    )

    # Array that stores objects detected
    object_detection = []

    for result in results:
        annotated_frame = result.plot()
        for box in result.boxes:
            class_index = int(box.cls)
            class_name = result.names[class_index]
            track_id = box.id
            # Add object detected to array
            object_detection.append((box, class_index, class_name, track_id))
            if track_id is None:
                continue
            for box, class_index, class_name, track_id in object_detection:
                track_id = int(track_id.item())
                if class_name == "cell phone":
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    y_mapped = map_object_to_screen(
                        y, 0, height, 0, screen_height, h, 100
                    )
                    center = (int(x + w / 2), int(y_mapped))
                    break

    return center, annotated_frame


def object_tracking(capture, screen_height):
    model = YOLO("yolo11s.pt")

    # Read the current frame (screen)
    ret, frame = capture.read()
    # Flip camera
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Unable to open the camera feed.")
        return None, (None, None), (None, None)

    # Get frame dimensions and split into halves
    height, width, _ = frame.shape
    left_frame = frame[:, :width // 2]
    right_frame = frame[:, width // 2:]

    # Initialize centers for each half
    center_left = (None, None)
    center_right = (None, None)

    # Process left half
    center_left, annotated_frame_left = process_frame_half(left_frame, model, height, screen_height)

    # Process right half
    center_right, annotated_frame_right = process_frame_half(right_frame, model, height, screen_height)

    # Combine the frames trough concat
    combined_frame = cv2.hconcat([left_frame, right_frame])
    combined_frame_annotated = cv2.hconcat([annotated_frame_left, annotated_frame_right])

    return combined_frame, center_left, center_right, combined_frame_annotated