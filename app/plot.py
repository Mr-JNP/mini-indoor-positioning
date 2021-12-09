import math
import cv2
import numpy as np


def bird_eye_view(frame, person_points, distance_w, distance_h, scale_w, scale_h):
    scale_w, scale_h = 1, 1
    h = frame.shape[0]
    w = frame.shape[1]

    white = (200, 200, 200)
    green = (0, 255, 0)

    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
    blank_image[:] = white

    scale_size_w = int(w * scale_w)
    scale_size_h = int(h * scale_h)

    scale_dis_w = int(distance_w * scale_w)
    scale_dis_h = int(distance_h * scale_h)

    cols = math.floor(scale_size_w / scale_dis_w)
    rows = math.floor(scale_size_h / scale_dis_h)

    for x in np.linspace(
        start=scale_dis_w, stop=scale_size_w - scale_dis_w, num=cols - 1
    ):
        x = int(round(x))
        cv2.line(blank_image, (x, 0), (x, scale_size_h), color=(0, 0, 0), thickness=2)

    # draw horizontal lines
    for y in np.linspace(
        start=scale_dis_h, stop=scale_size_h - scale_dis_h, num=rows - 1
    ):
        y = int(round(y))
        cv2.line(blank_image, (0, y), (scale_size_w, y), color=(0, 0, 0), thickness=1)

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
