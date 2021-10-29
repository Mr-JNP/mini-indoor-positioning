import cv2
import numpy as np


def bird_eye_view(frame, person_points, scale_w, scale_h):
    h = frame.shape[0]
    w = frame.shape[1]

    white = (200, 200, 200)
    green = (0, 255, 0)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white

    for i in person_points:
        blank_image = cv2.circle(
            blank_image, (int(i[0] * scale_w), int(i[1] * scale_h)), 5, green, 10
        )

    return blank_image
