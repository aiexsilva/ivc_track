import cv2
from ultralytics import YOLO
import logging

# Removes YOLO messages to the console
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def open_camera():
    capture = cv2.VideoCapture(0)
    return capture


def map_object_to_screen(y_obj, y_min_orig, y_max_orig, y_screen_min, y_screen_max, h_obj, h_barra):
    # Podes meter isto em ingles plis? foi o meu pai que escreveu e não me apetece trocar <3
    # Mapeamento linear do centro do objeto no espaço do ecrã
    y_screen = y_screen_min + (y_obj - y_min_orig) / (y_max_orig - y_min_orig) * (y_screen_max - y_screen_min)

    # Calcular o desfasamento proporcional à diferença de alturas
    desfasamento = max(0, (h_obj - h_barra) / 2)  # Garantir que o desfasamento não seja negativo
    fator_proporcional = (y_obj - y_min_orig) / (y_max_orig - y_min_orig)

    # Ajustar o desfasamento com base na posição do objeto no intervalo
    # Quando o objeto está no topo, o desfasamento reduz
    # Quando o objeto está no fundo, o desfasamento aumenta
    D = desfasamento * (2 * fator_proporcional - 1)

    # Ajustando a posição no ecrã consoante o desfasamento
    y_screen_adjusted = y_screen + D

    # Garantir que a posição ajustada fique dentro dos limites do ecrã
    y_screen_adjusted = max(y_screen_min, min(y_screen_adjusted, y_screen_max))

    return y_screen_adjusted


def process_frame_half(frame_half, model, height, screen_height):
    center = (None, None)

    results = model.track(
        frame_half,
        agnostic_nms=True,
        stream=True,
        persist=True,
        classes=[67],  # Class 67 = "cell phone"
        tracker="bytetrack.yaml"
    )

    for result in results:
        for box in result.boxes:
            class_index = int(box.cls)
            class_name = result.names[class_index]
            if class_name == "cell phone":
                x, y, w, h = box.xywh[0].cpu().numpy()
                y_mapped = map_object_to_screen(
                    y, 0, height, 0, screen_height, h, 100
                )
                center = (int(x + w / 2), int(y_mapped))
                break

    return center


def object_tracking(capture, screen_height):
    model = YOLO("yolo11s.pt")

    # Read the current frame (screen)
    ret, frame = capture.read()
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
    center_left = process_frame_half(left_frame, model, height, screen_height)

    # Process right half
    center_right = process_frame_half(right_frame, model, height, screen_height)

    # Combine the frames trough concat
    combined_frame = cv2.hconcat([left_frame, right_frame])

    return combined_frame, center_left, center_right
