from __future__ import annotations

import tempfile
from typing import Dict, List, Optional

import cv2
from datetime import datetime, timezone

# Lazy import MediaPipe to avoid import errors at module level
# Only import when actually needed (inside functions)
def _get_mediapipe():
    import mediapipe as mp
    return mp

def _get_pose_landmark():
    """Get PoseLandmark enum, only when needed."""
    mp = _get_mediapipe()
    return mp.solutions.pose.PoseLandmark


def _landmark_to_dict(lm) -> Dict:
    """Convert MediaPipe landmark to dictionary format."""
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(lm.visibility) if lm.HasField("visibility") else None,
    }


def _landmarks_to_list(landmarks) -> List[Dict]:
    """Convert MediaPipe landmarks list to list of dictionaries."""
    if not landmarks:
        return []
    return [_landmark_to_dict(lm) for lm in landmarks.landmark]


def _validate_hand_landmarks(hand_landmarks: List[Dict], min_visibility: float = 0.1) -> List[Dict]:
    """
    Validate and filter hand landmarks based on visibility.
    Hand landmarks should have 21 points. Filter out landmarks with very low visibility.
    
    Based on MediaPipe config: visibility_threshold: 0.1 for connections, 0.5 for joints.
    We use 0.1 as minimum to keep more landmarks but still filter out invalid ones.
    """
    if not hand_landmarks:
        return []
    
    # Hand landmarks should have exactly 21 points
    if len(hand_landmarks) != 21:
        # Return empty if count is wrong (might be corrupted data)
        return []
    
    validated = []
    for lm in hand_landmarks:
        # Hand landmarks don't have visibility field in MediaPipe, but we validate structure
        # Keep all landmarks but validate they have valid coordinates
        if lm.get("x") is not None and lm.get("y") is not None and lm.get("z") is not None:
            validated.append(lm)
    
    # If we lost landmarks, return empty (data might be corrupted)
    if len(validated) != 21:
        return []
    
    return validated


def extract_world_landmarks(
    video_path: str,
    stride: int = 1,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.65,  # Higher for better tracking stability
) -> Dict:
    """
    Process a video file and return holistic landmarks (pose, face, hands) per frame.

    Returns:
        {
            "frames": [
                {
                    "frame_index": int,
                    "poseLandmarks": [...],
                    "poseWorldLandmarks": [...],
                    "faceLandmarks": [...],
                    "leftHandLandmarks": [...],
                    "rightHandLandmarks": [...],
                },
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

    mp = _get_mediapipe()
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: List[Dict] = []
    # Build landmark indices mapping (only when needed)
    PoseLandmark = _get_pose_landmark()
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
            results = holistic.process(image_rgb)

            # Convert landmarks
            left_hand = _landmarks_to_list(results.left_hand_landmarks)
            right_hand = _landmarks_to_list(results.right_hand_landmarks)
            
            # Validate hand landmarks (ensure 21 points and valid coordinates)
            left_hand = _validate_hand_landmarks(left_hand)
            right_hand = _validate_hand_landmarks(right_hand)

            frame_data = {
                        "frame_index": idx,
                "poseLandmarks": _landmarks_to_list(results.pose_landmarks),
                "poseWorldLandmarks": _landmarks_to_list(results.pose_world_landmarks),
                "faceLandmarks": _landmarks_to_list(results.face_landmarks),
                "leftHandLandmarks": left_hand,
                "rightHandLandmarks": right_hand,
                        "segmentationMask": None,
                    }
            frames.append(frame_data)

            idx += 1
    finally:
        cap.release()
        holistic.close()

    first_frame = frames[0] if frames else {
        "poseLandmarks": [],
        "poseWorldLandmarks": [],
        "faceLandmarks": [],
        "leftHandLandmarks": [],
        "rightHandLandmarks": [],
    }
    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "metadata": {
          "timestamp": now_iso,
          "modelUrl": "https://threejs.org/examples/models/gltf/Michelle.glb",
          "exportVersion": "1.0",
          "source": "MediaPipe Holistic",
        },
        "poseLandmarks": first_frame.get("poseLandmarks", []),
        "poseWorldLandmarks": first_frame.get("poseWorldLandmarks", []),
        "faceLandmarks": first_frame.get("faceLandmarks", []),
        "leftHandLandmarks": first_frame.get("leftHandLandmarks", []),
        "rightHandLandmarks": first_frame.get("rightHandLandmarks", []),
        "segmentationMask": None,
        "landmarkIndices": landmark_indices,
        "frames": frames,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


