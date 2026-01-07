"""
Pose Comparator Module - Action Quality Assessment (AQA)

This module handles:
1. Feature Extraction: Convert XYZ coordinates to joint angles (Scale Invariant)
2. Temporal Alignment: Use Dynamic Time Warping (DTW) to sync different speeds
3. Weighted Scoring: Compare aligned frames with importance weights
4. Feedback Generation: Provide specific guidance on pose improvements

Author: Little MayXinh
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum


class MediaPipePoseIndex(IntEnum):
    """MediaPipe Pose Landmark indices for better readability."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class JointAngleDefinition:
    """Definition of a joint angle with 3 points and metadata."""
    name: str
    point_a: int  # First endpoint
    point_b: int  # Vertex (center point where angle is measured)
    point_c: int  # Second endpoint
    weight: float = 1.0  # Importance weight for scoring
    threshold_easy: float = 15.0  # Acceptable difference (degrees)
    threshold_medium: float = 25.0  # Noticeable difference
    threshold_hard: float = 40.0  # Significant difference


# Define yoga-relevant joint angles with weights
# Higher weight = more important for yoga poses
YOGA_JOINT_ANGLES: List[JointAngleDefinition] = [
    # Arms - Important for arm positions
    JointAngleDefinition(
        name="left_elbow",
        point_a=MediaPipePoseIndex.LEFT_SHOULDER,
        point_b=MediaPipePoseIndex.LEFT_ELBOW,
        point_c=MediaPipePoseIndex.LEFT_WRIST,
        weight=1.0,
    ),
    JointAngleDefinition(
        name="right_elbow",
        point_a=MediaPipePoseIndex.RIGHT_SHOULDER,
        point_b=MediaPipePoseIndex.RIGHT_ELBOW,
        point_c=MediaPipePoseIndex.RIGHT_WRIST,
        weight=1.0,
    ),
    
    # Legs - Critical for yoga stance
    JointAngleDefinition(
        name="left_knee",
        point_a=MediaPipePoseIndex.LEFT_HIP,
        point_b=MediaPipePoseIndex.LEFT_KNEE,
        point_c=MediaPipePoseIndex.LEFT_ANKLE,
        weight=1.5,  # Higher weight - legs are crucial
    ),
    JointAngleDefinition(
        name="right_knee",
        point_a=MediaPipePoseIndex.RIGHT_HIP,
        point_b=MediaPipePoseIndex.RIGHT_KNEE,
        point_c=MediaPipePoseIndex.RIGHT_ANKLE,
        weight=1.5,
    ),
    
    # Hip angles - Core posture
    JointAngleDefinition(
        name="left_hip",
        point_a=MediaPipePoseIndex.LEFT_SHOULDER,
        point_b=MediaPipePoseIndex.LEFT_HIP,
        point_c=MediaPipePoseIndex.LEFT_KNEE,
        weight=1.3,
    ),
    JointAngleDefinition(
        name="right_hip",
        point_a=MediaPipePoseIndex.RIGHT_SHOULDER,
        point_b=MediaPipePoseIndex.RIGHT_HIP,
        point_c=MediaPipePoseIndex.RIGHT_KNEE,
        weight=1.3,
    ),
    
    # Shoulder angles - Arm elevation
    JointAngleDefinition(
        name="left_shoulder",
        point_a=MediaPipePoseIndex.LEFT_ELBOW,
        point_b=MediaPipePoseIndex.LEFT_SHOULDER,
        point_c=MediaPipePoseIndex.LEFT_HIP,
        weight=1.0,
    ),
    JointAngleDefinition(
        name="right_shoulder",
        point_a=MediaPipePoseIndex.RIGHT_ELBOW,
        point_b=MediaPipePoseIndex.RIGHT_SHOULDER,
        point_c=MediaPipePoseIndex.RIGHT_HIP,
        weight=1.0,
    ),
    
    # Spine angles - Back straightness
    JointAngleDefinition(
        name="spine_upper",
        point_a=MediaPipePoseIndex.NOSE,
        point_b=MediaPipePoseIndex.LEFT_SHOULDER,  # Mid-shoulder approximation
        point_c=MediaPipePoseIndex.LEFT_HIP,
        weight=1.2,
    ),
    
    # Ankle angles - Foot position
    JointAngleDefinition(
        name="left_ankle",
        point_a=MediaPipePoseIndex.LEFT_KNEE,
        point_b=MediaPipePoseIndex.LEFT_ANKLE,
        point_c=MediaPipePoseIndex.LEFT_FOOT_INDEX,
        weight=0.8,
    ),
    JointAngleDefinition(
        name="right_ankle",
        point_a=MediaPipePoseIndex.RIGHT_KNEE,
        point_b=MediaPipePoseIndex.RIGHT_ANKLE,
        point_c=MediaPipePoseIndex.RIGHT_FOOT_INDEX,
        weight=0.8,
    ),
]


def calculate_angle_3d(
    point_a: Dict[str, float],
    point_b: Dict[str, float],
    point_c: Dict[str, float],
) -> float:
    """
    Calculate angle at point_b given 3D coordinates of three points.
    
    Args:
        point_a: First point with x, y, z coordinates
        point_b: Vertex point (where angle is measured)
        point_c: Third point with x, y, z coordinates
    
    Returns:
        Angle in degrees (0-180)
    """
    a = np.array([point_a['x'], point_a['y'], point_a['z']])
    b = np.array([point_b['x'], point_b['y'], point_b['z']])
    c = np.array([point_c['x'], point_c['y'], point_c['z']])
    
    # Vectors from vertex to endpoints
    ba = a - b
    bc = c - b
    
    # Handle zero-length vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-10 or norm_bc < 1e-10:
        return 0.0
    
    # Cosine of angle using dot product formula
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clamp to valid range for arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return float(angle)


def extract_pose_features(
    landmarks: List[Dict[str, float]],
    joint_definitions: List[JointAngleDefinition] = YOGA_JOINT_ANGLES,
) -> np.ndarray:
    """
    Convert raw pose landmarks into a feature vector of joint angles.
    This creates a "Pose Signature" that is scale-invariant.
    
    Args:
        landmarks: List of landmark dictionaries with x, y, z coordinates
        joint_definitions: List of joint angle definitions to extract
    
    Returns:
        NumPy array of angles (pose signature)
    """
    if not landmarks or len(landmarks) < 33:
        # Return zeros if invalid landmarks
        return np.zeros(len(joint_definitions))
    
    features = []
    
    for joint_def in joint_definitions:
        try:
            angle = calculate_angle_3d(
                landmarks[joint_def.point_a],
                landmarks[joint_def.point_b],
                landmarks[joint_def.point_c],
            )
            features.append(angle)
        except (IndexError, KeyError, TypeError):
            # If landmark is missing, use 0
            features.append(0.0)
    
    return np.array(features)


def extract_pose_sequence(frames: List[Dict]) -> List[np.ndarray]:
    """
    Extract pose features from all frames in a video.
    
    Args:
        frames: List of frame dictionaries containing poseWorldLandmarks
    
    Returns:
        List of feature vectors (pose signatures) for each frame
    """
    sequence = []
    
    for frame in frames:
        landmarks = frame.get('poseWorldLandmarks', [])
        features = extract_pose_features(landmarks)
        sequence.append(features)
    
    return sequence


def weighted_euclidean_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate weighted Euclidean distance between two pose vectors.
    
    Args:
        vec1: First pose feature vector
        vec2: Second pose feature vector
        weights: Optional weights for each dimension
    
    Returns:
        Weighted distance value
    """
    if weights is None:
        weights = np.array([j.weight for j in YOGA_JOINT_ANGLES])
    
    diff = vec1 - vec2
    weighted_diff = diff * weights
    
    return float(np.sqrt(np.sum(weighted_diff ** 2)))


def dynamic_time_warping(
    user_sequence: List[np.ndarray],
    trainer_sequence: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Perform Dynamic Time Warping (DTW) to align user and trainer sequences.
    
    This handles speed differences - if user is slower/faster than trainer,
    DTW finds the optimal alignment.
    
    Args:
        user_sequence: List of user pose feature vectors
        trainer_sequence: List of trainer pose feature vectors
        weights: Optional weights for distance calculation
    
    Returns:
        Tuple of (total_distance, alignment_path)
        - alignment_path: List of (user_index, trainer_index) pairs
    """
    n = len(user_sequence)
    m = len(trainer_sequence)
    
    if n == 0 or m == 0:
        return float('inf'), []
    
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = weighted_euclidean_distance(
                user_sequence[i - 1],
                trainer_sequence[j - 1],
                weights,
            )
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # Insertion
                dtw_matrix[i, j - 1],      # Deletion
                dtw_matrix[i - 1, j - 1],  # Match
            )
    
    # Backtrack to find the alignment path
    path = []
    i, j = n, m
    
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))  # 0-indexed
        
        # Find minimum of the three adjacent cells
        candidates = [
            (dtw_matrix[i - 1, j], i - 1, j),
            (dtw_matrix[i, j - 1], i, j - 1),
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    
    path.reverse()
    
    return float(dtw_matrix[n, m]), path


def calculate_pose_score(
    user_sequence: List[np.ndarray],
    trainer_sequence: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calculate overall pose quality score comparing user to trainer.
    
    Args:
        user_sequence: User's pose feature sequence
        trainer_sequence: Trainer's reference pose sequence
        weights: Optional joint importance weights
    
    Returns:
        Dictionary with score and analysis details
    """
    if not user_sequence or not trainer_sequence:
        return {
            "overall_score": 0.0,
            "dtw_distance": float('inf'),
            "average_angle_error": 0.0,
            "joint_scores": {},
        }
    
    # Get weights from joint definitions if not provided
    if weights is None:
        weights = np.array([j.weight for j in YOGA_JOINT_ANGLES])
    
    # Perform DTW alignment
    dtw_distance, path = dynamic_time_warping(
        user_sequence, trainer_sequence, weights
    )
    
    # Calculate per-joint errors along the alignment path
    joint_names = [j.name for j in YOGA_JOINT_ANGLES]
    joint_errors = {name: [] for name in joint_names}
    
    for user_idx, trainer_idx in path:
        user_vec = user_sequence[user_idx]
        trainer_vec = trainer_sequence[trainer_idx]
        
        for i, name in enumerate(joint_names):
            error = abs(user_vec[i] - trainer_vec[i])
            joint_errors[name].append(error)
    
    # Calculate average error per joint
    joint_avg_errors = {
        name: np.mean(errors) if errors else 0.0
        for name, errors in joint_errors.items()
    }
    
    # Calculate joint scores (0-100 scale)
    joint_scores = {}
    for i, joint_def in enumerate(YOGA_JOINT_ANGLES):
        avg_error = joint_avg_errors[joint_def.name]
        
        # Convert error to score using thresholds
        if avg_error <= joint_def.threshold_easy:
            # Perfect to good: 85-100
            score = 100 - (avg_error / joint_def.threshold_easy) * 15
        elif avg_error <= joint_def.threshold_medium:
            # Good to acceptable: 60-85
            progress = (avg_error - joint_def.threshold_easy) / (
                joint_def.threshold_medium - joint_def.threshold_easy
            )
            score = 85 - progress * 25
        elif avg_error <= joint_def.threshold_hard:
            # Acceptable to poor: 30-60
            progress = (avg_error - joint_def.threshold_medium) / (
                joint_def.threshold_hard - joint_def.threshold_medium
            )
            score = 60 - progress * 30
        else:
            # Poor: 0-30
            score = max(0, 30 - (avg_error - joint_def.threshold_hard) * 0.5)
        
        joint_scores[joint_def.name] = {
            "score": round(score, 1),
            "average_error_degrees": round(avg_error, 2),
            "weight": joint_def.weight,
        }
    
    # Calculate weighted overall score
    total_weight = sum(j.weight for j in YOGA_JOINT_ANGLES)
    weighted_score = sum(
        joint_scores[j.name]["score"] * j.weight
        for j in YOGA_JOINT_ANGLES
    ) / total_weight
    
    # Calculate average angle error
    average_angle_error = np.mean(list(joint_avg_errors.values()))
    
    return {
        "overall_score": round(weighted_score, 1),
        "dtw_distance": round(dtw_distance, 2),
        "average_angle_error": round(average_angle_error, 2),
        "alignment_length": len(path),
        "joint_scores": joint_scores,
    }


def generate_feedback(
    user_sequence: List[np.ndarray],
    trainer_sequence: List[np.ndarray],
    path: List[Tuple[int, int]],
    fps: float = 30.0,
    min_feedback_interval: float = 1.0,  # Minimum seconds between same feedback
) -> List[Dict]:
    """
    Generate human-readable feedback about pose corrections.
    
    Args:
        user_sequence: User's pose feature sequence
        trainer_sequence: Trainer's reference pose sequence
        path: DTW alignment path
        fps: Video frames per second
        min_feedback_interval: Minimum time between repeated feedback
    
    Returns:
        List of feedback items with timestamp and message
    """
    feedback = []
    last_feedback_time = {j.name: -min_feedback_interval for j in YOGA_JOINT_ANGLES}
    
    for user_idx, trainer_idx in path:
        timestamp = user_idx / fps
        user_vec = user_sequence[user_idx]
        trainer_vec = trainer_sequence[trainer_idx]
        
        for i, joint_def in enumerate(YOGA_JOINT_ANGLES):
            user_angle = user_vec[i]
            trainer_angle = trainer_vec[i]
            diff = user_angle - trainer_angle
            abs_diff = abs(diff)
            
            # Only generate feedback if error exceeds medium threshold
            # and enough time has passed since last feedback for this joint
            if abs_diff > joint_def.threshold_medium:
                if timestamp - last_feedback_time[joint_def.name] >= min_feedback_interval:
                    
                    # Generate specific guidance
                    message = _generate_joint_feedback(joint_def.name, diff, abs_diff)
                    
                    feedback.append({
                        "timestamp": round(timestamp, 2),
                        "joint": joint_def.name,
                        "message": message,
                        "error_degrees": round(abs_diff, 1),
                        "severity": _get_severity(abs_diff, joint_def),
                    })
                    
                    last_feedback_time[joint_def.name] = timestamp
    
    return feedback


def _generate_joint_feedback(joint_name: str, diff: float, abs_diff: float) -> str:
    """Generate specific feedback message for a joint."""
    
    # Determine direction
    need_more = diff > 0  # User's angle is larger = needs to bend/close more
    
    feedback_templates = {
        "left_elbow": {
            True: "Bend your left elbow more",
            False: "Straighten your left elbow",
        },
        "right_elbow": {
            True: "Bend your right elbow more",
            False: "Straighten your right elbow",
        },
        "left_knee": {
            True: "Bend your left knee more deeply",
            False: "Straighten your left leg more",
        },
        "right_knee": {
            True: "Bend your right knee more deeply",
            False: "Straighten your right leg more",
        },
        "left_hip": {
            True: "Lower your torso on the left side",
            False: "Lift your torso on the left side",
        },
        "right_hip": {
            True: "Lower your torso on the right side",
            False: "Lift your torso on the right side",
        },
        "left_shoulder": {
            True: "Lower your left arm",
            False: "Raise your left arm higher",
        },
        "right_shoulder": {
            True: "Lower your right arm",
            False: "Raise your right arm higher",
        },
        "spine_upper": {
            True: "Straighten your back",
            False: "Lean forward slightly",
        },
        "left_ankle": {
            True: "Point your left foot down more",
            False: "Flex your left foot up",
        },
        "right_ankle": {
            True: "Point your right foot down more",
            False: "Flex your right foot up",
        },
    }
    
    templates = feedback_templates.get(joint_name, {True: "Adjust position", False: "Adjust position"})
    return templates[need_more]


def _get_severity(error: float, joint_def: JointAngleDefinition) -> str:
    """Get severity level based on error magnitude."""
    if error <= joint_def.threshold_medium:
        return "low"
    elif error <= joint_def.threshold_hard:
        return "medium"
    else:
        return "high"


def compare_poses(
    trainer_frames: List[Dict],
    user_frames: List[Dict],
    trainer_fps: float = 30.0,
    user_fps: float = 30.0,
) -> Dict:
    """
    Main function to compare user's pose video against trainer's reference.
    
    This is the entry point for the comparison pipeline:
    1. Extract features from both sequences
    2. Perform DTW alignment
    3. Calculate scores
    4. Generate feedback
    
    Args:
        trainer_frames: Trainer's video frames with poseWorldLandmarks
        user_frames: User's video frames with poseWorldLandmarks
        trainer_fps: Trainer video FPS
        user_fps: User video FPS
    
    Returns:
        Complete comparison result with score and feedback
    """
    # Step 1: Extract pose features (Scale Invariant)
    trainer_sequence = extract_pose_sequence(trainer_frames)
    user_sequence = extract_pose_sequence(user_frames)
    
    if not trainer_sequence or not user_sequence:
        return {
            "success": False,
            "error": "Invalid pose data in video(s)",
            "overall_score": 0,
            "joint_scores": {},
            "feedback": [],
        }
    
    # Step 2: Calculate score with DTW alignment
    score_result = calculate_pose_score(user_sequence, trainer_sequence)
    
    # Step 3: Get DTW path for feedback generation
    _, path = dynamic_time_warping(user_sequence, trainer_sequence)
    
    # Step 4: Generate feedback
    feedback = generate_feedback(
        user_sequence,
        trainer_sequence,
        path,
        fps=user_fps,
    )
    
    return {
        "success": True,
        "overall_score": score_result["overall_score"],
        "dtw_distance": score_result["dtw_distance"],
        "average_angle_error": score_result["average_angle_error"],
        "joint_scores": score_result["joint_scores"],
        "feedback": feedback,
        "alignment_info": {
            "trainer_frames": len(trainer_sequence),
            "user_frames": len(user_sequence),
            "aligned_pairs": len(path),
        },
        "metadata": {
            "trainer_fps": trainer_fps,
            "user_fps": user_fps,
            "joints_analyzed": len(YOGA_JOINT_ANGLES),
        },
    }

