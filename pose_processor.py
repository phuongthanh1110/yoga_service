from __future__ import annotations

import tempfile
from typing import Dict, List

import cv2
import mediapipe as mp
from datetime import datetime, timezone

Holistic = mp.solutions.holistic.Holistic
PoseLandmark = mp.solutions.holistic.PoseLandmark
HandLandmark = mp.solutions.holistic.HandLandmark


def _landmarks_to_list(landmarks) -> List[Dict]:
    """Convert MediaPipe landmarks to a serialisable list of dicts."""
    if not landmarks:
        return []

    as_dicts: List[Dict] = []
    for lm in landmarks.landmark:
        visibility = None
        try:
            if hasattr(lm, "HasField") and lm.HasField("visibility"):
                visibility = float(lm.visibility)
        except ValueError:
            # Some landmark types do not define visibility; keep it None.
            visibility = None

        if visibility is None and getattr(lm, "visibility", None) is not None:
            visibility = float(lm.visibility)

        as_dicts.append(
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": visibility,
            }
        )
    return as_dicts


def _enum_index_map(enum_cls) -> Dict[int, str]:
    """Return {index: name} for MediaPipe enum classes."""
    return {i: member.name.lower() for i, member in enumerate(enum_cls)}


def _sequential_index_map(count: int, prefix: str) -> Dict[int, str]:
    """Fallback index map when enums are unavailable (e.g., face mesh)."""
    return {i: f"{prefix}_{i}" for i in range(count)}


def extract_world_landmarks(
    video_path: str,
    stride: int = 1,
    model_complexity: int = 1,  # Note: complexity=2 crashes on Apple Silicon
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

    holistic = Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=True,
        smooth_landmarks=True,  # Reduce jitter using temporal smoothing
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: List[Dict] = []
    pose_indices = _enum_index_map(PoseLandmark)
    hand_indices = _enum_index_map(HandLandmark)
    face_indices: Dict[int, str] = {}
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
            results = holistic.process(image_rgb)

            if results.pose_landmarks and results.pose_world_landmarks:
                img_landmarks = _landmarks_to_list(results.pose_landmarks)
                world_landmarks = _landmarks_to_list(results.pose_world_landmarks)
                left_hand = _landmarks_to_list(results.left_hand_landmarks)
                right_hand = _landmarks_to_list(results.right_hand_landmarks)
                left_hand_world = _landmarks_to_list(
                    getattr(results, "left_hand_world_landmarks", None)
                )
                right_hand_world = _landmarks_to_list(
                    getattr(results, "right_hand_world_landmarks", None)
                )
                face_landmarks = _landmarks_to_list(results.face_landmarks)
                if not face_indices and face_landmarks:
                    face_indices = _sequential_index_map(
                        len(face_landmarks), prefix="face"
                    )

                frames.append(
                    {
                        "frame_index": idx,
                        "poseLandmarks": img_landmarks,
                        "poseWorldLandmarks": world_landmarks,
                        "leftHandLandmarks": left_hand,
                        "rightHandLandmarks": right_hand,
                        "leftHandWorldLandmarks": left_hand_world,
                        "rightHandWorldLandmarks": right_hand_world,
                        "faceLandmarks": face_landmarks,
                        "segmentationMask": None,
                    }
                )

            idx += 1
    finally:
        cap.release()
        holistic.close()

    first_frame = frames[0] if frames else {"poseLandmarks": [], "poseWorldLandmarks": []}
    now_iso = datetime.now(timezone.utc).isoformat()

    pose_first = first_frame.get("poseLandmarks", [])
    pose_world_first = first_frame.get("poseWorldLandmarks", [])
    left_hand_first = first_frame.get("leftHandLandmarks", [])
    right_hand_first = first_frame.get("rightHandLandmarks", [])
    left_hand_world_first = first_frame.get("leftHandWorldLandmarks", [])
    right_hand_world_first = first_frame.get("rightHandWorldLandmarks", [])
    face_first = first_frame.get("faceLandmarks", [])
    face_indices = face_indices or _sequential_index_map(len(face_first), prefix="face")

    return {
        "metadata": {
            "timestamp": now_iso,
            "modelUrl": "https://threejs.org/examples/models/gltf/Michelle.glb",
            "exportVersion": "1.0",
            "source": "MediaPipe Holistic",
        },
        "poseLandmarks": pose_first,
        "poseWorldLandmarks": pose_world_first,
        "leftHandLandmarks": left_hand_first,
        "rightHandLandmarks": right_hand_first,
        "leftHandWorldLandmarks": left_hand_world_first,
        "rightHandWorldLandmarks": right_hand_world_first,
        "faceLandmarks": face_first,
        "segmentationMask": None,
        "landmarkIndices": pose_indices,
        "handLandmarkIndices": hand_indices,
        "faceLandmarkIndices": face_indices,
        "frames": frames,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


