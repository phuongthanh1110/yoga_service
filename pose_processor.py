from __future__ import annotations

import tempfile
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
from datetime import datetime, timezone

PoseLandmark = mp.solutions.pose.PoseLandmark


def extract_world_landmarks(
    video_path: str,
    stride: int = 1,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Dict:
    """
    Process a video file and return world landmarks per frame.

    Returns:
        {
            "frames": [
                {"frame_index": int, "landmarks": [{"x":..., "y":..., "z":..., "visibility":...}, ...]},
            ],
            "frame_count": int,
            "fps": float,
            "width": int,
            "height": int,
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: List[Dict] = []
    landmark_indices = {i: name.name.lower() for i, name in enumerate(PoseLandmark)}
    idx = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if stride > 1 and (idx % stride) != 0:
                idx += 1
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks and results.pose_world_landmarks:
                img_landmarks = []
                for lm in results.pose_landmarks.landmark:
                    img_landmarks.append(
                        {
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility) if lm.HasField("visibility") else None,
                        }
                    )

                world_landmarks = []
                for lm in results.pose_world_landmarks.landmark:
                    world_landmarks.append(
                        {
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility) if lm.HasField("visibility") else None,
                        }
                    )

                frames.append(
                    {
                        "frame_index": idx,
                        "poseLandmarks": img_landmarks,
                        "poseWorldLandmarks": world_landmarks,
                        "segmentationMask": None,
                    }
                )

            idx += 1
    finally:
        cap.release()
        pose.close()

    first_frame = frames[0] if frames else {"poseLandmarks": [], "poseWorldLandmarks": []}
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "metadata": {
          "timestamp": now_iso,
          "modelUrl": "https://threejs.org/examples/models/gltf/Michelle.glb",
          "exportVersion": "1.0",
          "source": "MediaPipe Pose",
        },
        "poseLandmarks": first_frame.get("poseLandmarks", []),
        "poseWorldLandmarks": first_frame.get("poseWorldLandmarks", []),
        "segmentationMask": None,
        "landmarkIndices": landmark_indices,
        "frames": frames,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


