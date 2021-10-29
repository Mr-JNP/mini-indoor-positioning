import cv2
import numpy as np


def get_transformed_points(boxes, prespective_transform):
    bottom_points = []
    for box in boxes:
        pnts = np.array(
            [[[int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]]], dtype="float32"
        )
        bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        bottom_points.append(pnt)

    return bottom_points


def get_scale(W, H, dis_w=600, dis_h=600):
    return float(dis_w / W), float(dis_h / H)
