"""Extract wrist positions and pinch distances from a two-handed video.

Reads a video frame by frame, runs MediaPipe's HandLandmarker on each frame,
and writes one row per frame to a CSV: where each wrist was, and how far apart
the thumb and index fingertip were (the "pinch" signal that will drive the
robot's gripper later).

Run this inside the mp-env venv, NOT lerobot-env.

    python day12/track_hands.py --max-frames 60      # smoke test, a few seconds
    python day12/track_hands.py                      # full pass, a few minutes
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

matplotlib.use("Agg")  # no display in WSL; render straight to file
import matplotlib.pyplot as plt  # noqa: E402

# MediaPipe hand landmark indices. There are 21 per hand; these are the three
# we care about. See the hand landmark diagram in MediaPipe's docs.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8

CSV_COLUMNS = [
    "frame_index",
    "time_seconds",
    "num_hands",
    "left_wrist_x",
    "left_wrist_y",
    "left_pinch_m",
    "right_wrist_x",
    "right_wrist_y",
    "right_pinch_m",
]


def build_landmarker(model_path, detection_confidence, tracking_confidence):
    """Create the MediaPipe hand tracker in VIDEO mode.

    VIDEO mode (as opposed to IMAGE mode) lets MediaPipe carry tracking state
    between frames, which is both faster and steadier than detecting from
    scratch every frame. It requires timestamps that always increase.
    """
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,  # defaults to 1, which would silently drop a hand
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def pinch_distance_metres(world_landmarks):
    """Distance from thumb tip to index fingertip, in metres.

    Uses MediaPipe's *world* landmarks, which are metric and centred on the
    hand itself. That makes this measurement independent of how close the hand
    was to the camera -- a raw pixel distance would shrink whenever the hand
    moved away from the lens, and no single threshold would work.
    """
    thumb = world_landmarks[THUMB_TIP]
    index = world_landmarks[INDEX_TIP]
    return math.sqrt(
        (thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2 + (thumb.z - index.z) ** 2
    )


def extract_hands(detection_result):
    """Pull the numbers we care about out of MediaPipe's result object.

    Returns a list of dicts, one per detected hand, sorted left-to-right by
    where the wrist sits in the image.
    """
    hands = []
    for hand_index in range(len(detection_result.hand_landmarks)):
        wrist = detection_result.hand_landmarks[hand_index][WRIST]
        hands.append(
            {
                # Normalized image coordinates, 0..1. x=0 is the left edge of
                # the frame, y=0 is the TOP edge (image y grows downward).
                "wrist_x": wrist.x,
                "wrist_y": wrist.y,
                "pinch_m": pinch_distance_metres(
                    detection_result.hand_world_landmarks[hand_index]
                ),
            }
        )
    hands.sort(key=lambda hand: hand["wrist_x"])
    return hands


def assign_sides(hands, last_left_x, last_right_x):
    """Decide which detected hand is the left-in-image one and which is right.

    We deliberately ignore MediaPipe's own handedness label. That label assumes
    a mirrored selfie-view image; this footage is unmirrored rear-camera, so the
    label is inverted and untrustworthy. Image position is unambiguous instead.

    With two hands it is trivial (already sorted by x). With one hand we guess
    from where each hand was last seen, falling back to a midpoint split.
    """
    if len(hands) >= 2:
        return hands[0], hands[1]

    if len(hands) == 0:
        return None, None

    only_hand = hands[0]
    if last_left_x is not None and last_right_x is not None:
        distance_to_left = abs(only_hand["wrist_x"] - last_left_x)
        distance_to_right = abs(only_hand["wrist_x"] - last_right_x)
        if distance_to_left <= distance_to_right:
            return only_hand, None
        return None, only_hand

    if only_hand["wrist_x"] < 0.5:
        return only_hand, None
    return None, only_hand


def row_value(hand, key):
    """CSV cell for one measurement, or 'nan' when the hand wasn't detected.

    We write a row for every single frame even when tracking failed, so the
    time axis stays honest and gaps stay visible instead of silently closing up.
    """
    if hand is None:
        return "nan"
    return f"{hand[key]:.6f}"


def annotate_frame(frame, hands):
    """Draw the tracked points onto a copy of the frame, for eyeballing."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    colours = [(0, 255, 0), (0, 165, 255)]  # BGR: green for first, orange for second
    for hand_index, hand in enumerate(hands):
        colour = colours[hand_index % len(colours)]
        centre = (int(hand["wrist_x"] * width), int(hand["wrist_y"] * height))
        cv2.circle(annotated, centre, 14, colour, 3)
        cv2.putText(
            annotated,
            f"pinch {hand['pinch_m'] * 100:.1f}cm",
            (centre[0] + 20, centre[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colour,
            2,
        )
    return annotated


def save_plot(rows, plot_path):
    """Two-panel sanity check: wrist paths, and pinch signal over time."""
    times = np.array([row[1] for row in rows], dtype=float)
    left_x = np.array([row[3] for row in rows], dtype=float)
    left_y = np.array([row[4] for row in rows], dtype=float)
    left_pinch = np.array([row[5] for row in rows], dtype=float)
    right_x = np.array([row[6] for row in rows], dtype=float)
    right_y = np.array([row[7] for row in rows], dtype=float)
    right_pinch = np.array([row[8] for row in rows], dtype=float)

    figure, (path_axis, pinch_axis) = plt.subplots(1, 2, figsize=(14, 6))

    path_axis.plot(left_x, left_y, ".-", markersize=2, linewidth=0.6, label="left-in-image")
    path_axis.plot(right_x, right_y, ".-", markersize=2, linewidth=0.6, label="right-in-image")
    path_axis.set_xlim(0, 1)
    path_axis.set_ylim(1, 0)  # flip: image y grows downward, so this reads like the video
    path_axis.set_xlabel("image x (0 = left edge)")
    path_axis.set_ylabel("image y (0 = top edge)")
    path_axis.set_title("wrist paths, normalized image coordinates")
    path_axis.legend()
    path_axis.grid(alpha=0.3)

    pinch_axis.plot(times, left_pinch * 100, linewidth=0.9, label="left-in-image")
    pinch_axis.plot(times, right_pinch * 100, linewidth=0.9, label="right-in-image")
    pinch_axis.set_xlabel("time (seconds)")
    pinch_axis.set_ylabel("thumb-to-index distance (cm)")
    pinch_axis.set_title("pinch signal -- dips are grasps")
    pinch_axis.legend()
    pinch_axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=120)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="day12/hand_video.MOV")
    parser.add_argument("--model", default="day12/hand_landmarker.task")
    parser.add_argument("--out-csv", default="day12/hand_trajectory.csv")
    parser.add_argument("--out-plot", default="day12/wrist_paths.png")
    parser.add_argument("--out-check-frame", default="day12/track_check.jpg")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after this many frames (0 = whole video). Use a small number to smoke test.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="skip this many frames before tracking. Smoke test the middle of the clip, "
        "not the first second -- hands are often not in frame yet at the start.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="resize frames by this factor before detection. MediaPipe's palm detector "
        "can struggle when hands fill a 1080p frame; try 0.5.",
    )
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    arguments = parser.parse_args()

    video_path = Path(arguments.video)
    model_path = Path(arguments.model)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open {video_path}")

    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    if not frames_per_second or frames_per_second <= 0:
        frames_per_second = 60.0
        print("warning: could not read fps from file, assuming 60")
    print(f"video: {video_path}  fps={frames_per_second:.3f}")

    landmarker = build_landmarker(
        model_path, arguments.detection_confidence, arguments.tracking_confidence
    )

    rows = []
    last_left_x = None
    last_right_x = None
    check_frame_saved = False
    hand_count_tally = {0: 0, 1: 0, 2: 0}
    frame_index = 0
    processed_count = 0

    while True:
        got_frame, frame_bgr = capture.read()
        if not got_frame:
            break

        # Decode-and-discard until the requested start. Cheap compared to
        # inference, and more reliable across codecs than seeking by index.
        if frame_index < arguments.start_frame:
            frame_index += 1
            continue
        if arguments.max_frames and processed_count >= arguments.max_frames:
            break

        if arguments.scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, None, fx=arguments.scale, fy=arguments.scale,
                interpolation=cv2.INTER_AREA,
            )

        # MediaPipe wants RGB; OpenCV hands us BGR. Skipping this conversion
        # does not error -- it just quietly makes detection much worse.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Derive the timestamp from the frame index rather than asking OpenCV
        # for a position, which can repeat a value and break VIDEO mode's
        # requirement that timestamps strictly increase.
        timestamp_ms = int(frame_index / frames_per_second * 1000)
        detection_result = landmarker.detect_for_video(mediapipe_image, timestamp_ms)

        hands = extract_hands(detection_result)
        hand_count_tally[min(len(hands), 2)] += 1
        left_hand, right_hand = assign_sides(hands, last_left_x, last_right_x)
        if left_hand is not None:
            last_left_x = left_hand["wrist_x"]
        if right_hand is not None:
            last_right_x = right_hand["wrist_x"]

        rows.append(
            [
                frame_index,
                f"{frame_index / frames_per_second:.6f}",
                len(hands),
                row_value(left_hand, "wrist_x"),
                row_value(left_hand, "wrist_y"),
                row_value(left_hand, "pinch_m"),
                row_value(right_hand, "wrist_x"),
                row_value(right_hand, "wrist_y"),
                row_value(right_hand, "pinch_m"),
            ]
        )

        # Save one annotated frame from the first moment both hands are seen,
        # so there is a visual record that tracking latched onto real hands.
        if not check_frame_saved and len(hands) == 2:
            cv2.imwrite(arguments.out_check_frame, annotate_frame(frame_bgr, hands))
            check_frame_saved = True
            print(f"wrote check frame from frame {frame_index} -> {arguments.out_check_frame}")

        frame_index += 1
        processed_count += 1
        if processed_count % 200 == 0:
            print(f"  ...{processed_count} frames processed")

    capture.release()

    with open(arguments.out_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    save_plot(rows, arguments.out_plot)

    total = len(rows)
    both = hand_count_tally[2]
    print()
    print(f"frames processed   : {total}")
    print(f"two hands detected : {both} ({100.0 * both / max(total, 1):.1f}%)")
    print(f"one hand detected  : {hand_count_tally[1]}")
    print(f"no hands detected  : {hand_count_tally[0]}")
    print(f"csv  -> {arguments.out_csv}")
    print(f"plot -> {arguments.out_plot}")
    if not check_frame_saved:
        print("WARNING: never saw two hands at once -- check the video and confidences")


if __name__ == "__main__":
    main()
