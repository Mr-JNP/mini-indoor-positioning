import os
import argparse
import cv2
import json
import numpy as np
import pandas as pd

from app.plot import bird_eye_view
from app.utils import get_scale, get_transformed_points

mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(
                image,
                (x, y),
                (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]),
                (70, 70, 70),
                2,
            )
            if len(mouse_pts) == 3:
                cv2.line(
                    image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2
                )
        mouse_pts.append((x, y))


def transform_to_floor_plan_view(video_path, bb_path, output_vid, output_dir="output"):
    assert video_path is None or os.path.isfile(video_path), "{} is not a file".format(
        video_path
    )
    assert bb_path is None or os.path.isfile(bb_path), "{} is not a file".format(
        bb_path
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_path = os.path.join(output_dir, output_vid)
    json_output_path = f'{result_path.split(".")[0]}.json'

    vs = cv2.VideoCapture(video_path)

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = get_scale(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    bird_movie = cv2.VideoWriter(
        result_path,
        fourcc,
        fps,
        (int(width * scale_w), int(height * scale_h)),
    )

    with open(bb_path, "rb") as f:
        bb = np.load(f, allow_pickle=True)
    bb_df = pd.DataFrame(bb[:, 0:6])
    bb_df_grouped = bb_df.groupby(0)
    frame_ids = bb_df_grouped.groups.keys()

    frame_id = 0
    points = []
    positions = []
    global image

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        (H, W) = frame.shape[:2]

        if frame_id == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break
            points = mouse_pts
            src = np.float32(np.array(points[:4]))
            dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
            perspective_transform = cv2.getPerspectiveTransform(src, dst)

            # using next 3 points for horizontal and vertical unit length(in this case 100 cm)
            pts = np.float32(np.array([points[4:7]]))
            warped_pt = cv2.perspectiveTransform(pts, perspective_transform)[0]

            distance_w = np.sqrt(
                (warped_pt[0][0] - warped_pt[1][0]) ** 2
                + (warped_pt[0][1] - warped_pt[1][1]) ** 2
            )
            distance_h = np.sqrt(
                (warped_pt[0][0] - warped_pt[2][0]) ** 2
                + (warped_pt[0][1] - warped_pt[2][1]) ** 2
            )

            cv2.imwrite(os.path.join(output_dir, "ref.png"), image)

        boxes = []
        track = []
        if frame_id in frame_ids:
            boxes = bb_df_grouped.get_group(frame_id).values[:, 2:6]
            track = (
                bb_df_grouped.get_group(frame_id).iloc[:, 1].astype(np.int64).to_list()
            )

        if frame_id != 0:
            point_for_vis = []
            person_points = get_transformed_points(boxes, perspective_transform)

            for i, k in enumerate(track):
                person_point = person_points[i]
                point_for_vis.append([frame_id, k, *person_point])

                coordinate = [
                    int(person_point[0] * scale_w),
                    int(person_point[1] * scale_h),
                ]
                pos = {"frame_id": frame_id, "track_id": k, "position": coordinate}
                if frame_id % 6 == 0:
                    positions.append(pos)

            bird_image = bird_eye_view(
                frame, point_for_vis, distance_w, distance_h, scale_w, scale_h
            )
            bird_movie.write(bird_image)

            if frame_id == 1:
                inv_bird_image = cv2.warpPerspective(
                    bird_image,
                    np.linalg.inv(perspective_transform),
                    (W, H),
                )
                cv2.imwrite(os.path.join(output_dir, "transformed.png"), inv_bird_image)

        frame_id = frame_id + 1

    vs.release()
    cv2.destroyAllWindows()

    with open(json_output_path, "w", encoding="utf8") as json_file:
        json_data = {"video": video_path, "positions": positions}
        json.dump(json_data, json_file, ensure_ascii=False)


if __name__ == "__main__":

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v",
        "--video_path",
        action="store",
        dest="video_path",
        default="./data/example.mp4",
        help="Path for input video",
    )

    parser.add_argument(
        "-b",
        "--bounding_boxes",
        action="store",
        dest="bounding_boxes",
        default="./data/example.npy",
        help="Path for bounding boxes",
    )

    parser.add_argument(
        "-O",
        "--output_vid",
        action="store",
        dest="output_vid",
        default="example.avi",
        help="Path for Output videos",
    )

    args = parser.parse_args()
    video_path = args.video_path
    bb_path = args.bounding_boxes
    output_vid = args.output_vid

    # set mouse callback
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)

    transform_to_floor_plan_view(video_path, bb_path, output_vid)
