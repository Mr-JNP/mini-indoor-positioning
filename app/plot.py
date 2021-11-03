import cv2
import numpy as np


def bird_eye_view(frame, person_points, scale_w, scale_h):
    h = frame.shape[0]
    w = frame.shape[1]

    white = (200, 200, 200)
    green = (0, 255, 0)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white

    text_scale = max(1, w / 1600.0)
    text_thickness = 2

    for _, track_id, x, y in person_points:
        blank_image = cv2.circle(
            blank_image, (int(x * scale_w), int(y * scale_h)), 5, green, 10
        )
        blank_image = cv2.putText(
            blank_image,
            str(track_id),
            (int(x * scale_w), int(y * scale_h) - 30),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (255, 0, 0),
            thickness=text_thickness,
        )

    return blank_image
