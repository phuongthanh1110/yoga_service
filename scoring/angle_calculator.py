"""Calculate angles between joints from landmarks."""

import math
from typing import List, Optional, Tuple


def calculate_angle(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    point3: Tuple[float, float, float],
) -> float:
    """
    Calculate angle at point2 formed by point1-point2-point3.
    
    Args:
        point1: First point (x, y, z)
        point2: Vertex point (x, y, z)
        point3: Third point (x, y, z)
    
    Returns:
        Angle in degrees (0-180)
    """
    # Convert to vectors
    vec1 = (
        point1[0] - point2[0],
        point1[1] - point2[1],
        point1[2] - point2[2],
    )
    vec2 = (
        point3[0] - point2[0],
        point3[1] - point2[1],
        point3[2] - point2[2],
    )
    
    # Calculate dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    
    # Calculate magnitudes
    mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2 + vec1[2] ** 2)
    mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2 + vec2[2] ** 2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle in radians, then convert to degrees
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to avoid domain errors
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def get_landmark_point(landmarks: List[dict], index: int) -> Optional[Tuple[float, float, float]]:
    """
    Extract point coordinates from landmarks list.
    
    Args:
        landmarks: List of landmark dicts with x, y, z
        index: Landmark index
    
    Returns:
        Tuple (x, y, z) or None if invalid
    """
    if not landmarks or index < 0 or index >= len(landmarks):
        return None
    
    landmark = landmarks[index]
    if not isinstance(landmark, dict):
        return None
    
    x = landmark.get("x")
    y = landmark.get("y")
    z = landmark.get("z")
    
    if x is None or y is None or z is None:
        return None
    
    return (float(x), float(y), float(z))


def calculate_joint_angles(landmarks: List[dict]) -> dict:
    """
    Calculate key joint angles from pose landmarks.
    
    Uses MediaPipe Pose landmark indices:
    - 11: Left shoulder
    - 13: Left elbow
    - 15: Left wrist
    - 12: Right shoulder
    - 14: Right elbow
    - 16: Right wrist
    - 23: Left hip
    - 25: Left knee
    - 27: Left ankle
    - 24: Right hip
    - 26: Right knee
    - 28: Right ankle
    
    Returns:
        Dict with angle names and values in degrees
    """
    angles = {}
    
    # Left arm angles
    left_shoulder = get_landmark_point(landmarks, 11)
    left_elbow = get_landmark_point(landmarks, 13)
    left_wrist = get_landmark_point(landmarks, 15)
    left_hip = get_landmark_point(landmarks, 23)
    
    if left_shoulder and left_elbow and left_wrist:
        angles["left_elbow"] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    if left_hip and left_shoulder and left_elbow:
        angles["left_shoulder"] = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Right arm angles
    right_shoulder = get_landmark_point(landmarks, 12)
    right_elbow = get_landmark_point(landmarks, 14)
    right_wrist = get_landmark_point(landmarks, 16)
    right_hip = get_landmark_point(landmarks, 24)
    
    if right_shoulder and right_elbow and right_wrist:
        angles["right_elbow"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    if right_hip and right_shoulder and right_elbow:
        angles["right_shoulder"] = calculate_angle(right_hip, right_shoulder, right_elbow)
    
    # Left leg angles
    left_knee = get_landmark_point(landmarks, 25)
    left_ankle = get_landmark_point(landmarks, 27)
    
    if left_hip and left_knee and left_ankle:
        angles["left_knee"] = calculate_angle(left_hip, left_knee, left_ankle)
    
    # Right leg angles
    right_knee = get_landmark_point(landmarks, 26)
    right_ankle = get_landmark_point(landmarks, 28)
    
    if right_hip and right_knee and right_ankle:
        angles["right_knee"] = calculate_angle(right_hip, right_knee, right_ankle)
    
    return angles

