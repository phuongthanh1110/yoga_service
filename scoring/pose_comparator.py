"""Compare two poses and calculate differences."""

from typing import List, Dict, Optional
from .angle_calculator import calculate_joint_angles, get_landmark_point


class PoseComparison:
    """Result of comparing two poses."""
    
    def __init__(self):
        self.angle_differences: Dict[str, float] = {}
        self.position_differences: Dict[str, float] = {}
        self.overall_angle_error: float = 0.0
        self.overall_position_error: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "angle_differences": self.angle_differences,
            "position_differences": self.position_differences,
            "overall_angle_error": self.overall_angle_error,
            "overall_position_error": self.overall_position_error,
        }


def calculate_euclidean_distance(
    point1: tuple,
    point2: tuple,
) -> float:
    """Calculate Euclidean distance between two 3D points."""
    if point1 is None or point2 is None:
        return float('inf')
    
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    dz = point1[2] - point2[2]
    
    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5


def normalize_by_scale(
    landmarks: List[dict],
    scale_landmark_indices: tuple = (11, 12),  # Left and right shoulders
) -> Optional[float]:
    """
    Calculate normalization scale based on body size.
    Uses distance between shoulders as reference.
    
    Returns:
        Scale factor or None if cannot calculate
    """
    if len(landmarks) < max(scale_landmark_indices) + 1:
        return None
    
    point1 = get_landmark_point(landmarks, scale_landmark_indices[0])
    point2 = get_landmark_point(landmarks, scale_landmark_indices[1])
    
    if point1 is None or point2 is None:
        return None
    
    distance = calculate_euclidean_distance(point1, point2)
    if distance == 0:
        return None
    
    return distance


def compare_poses(
    reference_landmarks: List[dict],
    user_landmarks: List[dict],
    normalize: bool = True,
) -> PoseComparison:
    """
    Compare reference pose with user pose.
    
    Args:
        reference_landmarks: Reference pose landmarks
        user_landmarks: User pose landmarks
        normalize: Whether to normalize by body size
    
    Returns:
        PoseComparison object with differences
    """
    comparison = PoseComparison()
    
    # Calculate angles for both poses
    ref_angles = calculate_joint_angles(reference_landmarks)
    user_angles = calculate_joint_angles(user_landmarks)
    
    # Compare angles
    angle_errors = []
    for joint_name in ref_angles:
        if joint_name in user_angles:
            ref_angle = ref_angles[joint_name]
            user_angle = user_angles[joint_name]
            diff = abs(ref_angle - user_angle)
            comparison.angle_differences[joint_name] = diff
            angle_errors.append(diff)
    
    if angle_errors:
        comparison.overall_angle_error = sum(angle_errors) / len(angle_errors)
    
    # Compare positions of key landmarks
    key_landmark_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    # Shoulders, elbows, wrists, hips, knees, ankles
    
    # Normalize by body size if requested
    scale_factor = 1.0
    if normalize:
        ref_scale = normalize_by_scale(reference_landmarks)
        user_scale = normalize_by_scale(user_landmarks)
        if ref_scale and user_scale and ref_scale > 0:
            scale_factor = ref_scale / user_scale
    
    position_errors = []
    for idx in key_landmark_indices:
        ref_point = get_landmark_point(reference_landmarks, idx)
        user_point = get_landmark_point(user_landmarks, idx)
        
        if ref_point and user_point:
            # Scale user point to match reference scale
            if normalize and scale_factor != 1.0:
                user_point = (
                    user_point[0] * scale_factor,
                    user_point[1] * scale_factor,
                    user_point[2] * scale_factor,
                )
            
            distance = calculate_euclidean_distance(ref_point, user_point)
            landmark_name = f"landmark_{idx}"
            comparison.position_differences[landmark_name] = distance
            position_errors.append(distance)
    
    if position_errors:
        comparison.overall_position_error = sum(position_errors) / len(position_errors)
    
    return comparison

