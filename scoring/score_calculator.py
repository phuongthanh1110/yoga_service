"""Calculate overall score from pose comparisons."""

from typing import List, Dict
from .pose_comparator import compare_poses, PoseComparison


class ScoreResult:
    """Overall scoring result."""
    
    def __init__(self):
        self.overall_score: float = 0.0
        self.angle_accuracy: float = 0.0
        self.position_accuracy: float = 0.0
        self.stability_score: float = 0.0
        self.frame_scores: List[float] = []
        self.feedback: List[Dict] = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": round(self.overall_score, 2),
            "angle_accuracy": round(self.angle_accuracy, 2),
            "position_accuracy": round(self.position_accuracy, 2),
            "stability_score": round(self.stability_score, 2),
            "frame_scores": [round(score, 2) for score in self.frame_scores],
            "feedback": self.feedback,
        }


def calculate_frame_score(comparison: PoseComparison) -> float:
    """
    Calculate score for a single frame comparison.
    
    Score formula:
    - Angle error: 0-180 degrees -> 0-100 points (linear)
    - Position error: normalized -> 0-100 points
    - Combined: weighted average
    
    Returns:
        Score from 0-100
    """
    # Angle score: 0 degrees error = 100, 30 degrees error = 0
    max_angle_error = 30.0
    angle_error = comparison.overall_angle_error
    angle_score = max(0.0, 100.0 * (1.0 - angle_error / max_angle_error))
    
    # Position score: normalize error (assuming normalized landmarks)
    # Typical good position error < 0.1, bad > 0.5
    max_position_error = 0.5
    position_error = comparison.overall_position_error
    position_score = max(0.0, 100.0 * (1.0 - position_error / max_position_error))
    
    # Weighted combination: 70% angle, 30% position
    frame_score = angle_score * 0.7 + position_score * 0.3
    
    return min(100.0, max(0.0, frame_score))


def calculate_stability_score(frame_scores: List[float]) -> float:
    """
    Calculate stability score based on consistency.
    
    Uses standard deviation: lower deviation = higher stability.
    
    Returns:
        Stability score from 0-100
    """
    if not frame_scores:
        return 0.0
    
    if len(frame_scores) == 1:
        return 100.0
    
    # Calculate mean and standard deviation
    mean_score = sum(frame_scores) / len(frame_scores)
    
    variance = sum((score - mean_score) ** 2 for score in frame_scores) / len(frame_scores)
    std_dev = variance ** 0.5
    
    # Stability: lower std_dev = higher score
    # Max std_dev for 0 score: 50 (very inconsistent)
    max_std_dev = 50.0
    stability = max(0.0, 100.0 * (1.0 - std_dev / max_std_dev))
    
    return min(100.0, stability)


def calculate_overall_score(
    reference_frames: List[dict],
    user_frames: List[dict],
    reference_fps: float = 30.0,
    user_fps: float = 30.0,
) -> ScoreResult:
    """
    Calculate overall score by comparing all frames.
    
    Args:
        reference_frames: Reference pose frames
        user_frames: User pose frames
        reference_fps: FPS of reference
        user_fps: FPS of user
    
    Returns:
        ScoreResult with all metrics
    """
    from .time_aligner import align_pose_sequences
    
    result = ScoreResult()
    
    # Align sequences
    aligned_pairs = align_pose_sequences(
        reference_frames,
        user_frames,
        reference_fps,
        user_fps,
    )
    
    if not aligned_pairs:
        return result
    
    # Compare each aligned pair
    comparisons: List[PoseComparison] = []
    frame_scores: List[float] = []
    
    for ref_frame, user_frame in aligned_pairs:
        ref_landmarks = ref_frame.get("poseWorldLandmarks", [])
        user_landmarks = user_frame.get("poseWorldLandmarks", [])
        
        if not ref_landmarks or not user_landmarks:
            continue
        
        comparison = compare_poses(ref_landmarks, user_landmarks, normalize=True)
        comparisons.append(comparison)
        
        frame_score = calculate_frame_score(comparison)
        frame_scores.append(frame_score)
    
    if not frame_scores:
        return result
    
    # Calculate overall metrics
    result.frame_scores = frame_scores
    result.overall_score = sum(frame_scores) / len(frame_scores)
    
    # Angle accuracy: average of all angle comparisons
    if comparisons:
        angle_errors = [c.overall_angle_error for c in comparisons]
        avg_angle_error = sum(angle_errors) / len(angle_errors)
        result.angle_accuracy = max(0.0, 100.0 * (1.0 - avg_angle_error / 30.0))
    
    # Position accuracy: average of all position comparisons
    if comparisons:
        position_errors = [c.overall_position_error for c in comparisons]
        avg_position_error = sum(position_errors) / len(position_errors)
        result.position_accuracy = max(0.0, 100.0 * (1.0 - avg_position_error / 0.5))
    
    # Stability score
    result.stability_score = calculate_stability_score(frame_scores)
    
    # Generate feedback for worst frames
    if frame_scores:
        worst_indices = sorted(
            range(len(frame_scores)),
            key=lambda i: frame_scores[i],
        )[:3]  # Top 3 worst frames
        
        for idx in worst_indices:
            if idx < len(comparisons):
                comp = comparisons[idx]
                worst_joint = max(
                    comp.angle_differences.items(),
                    key=lambda x: x[1],
                    default=("unknown", 0.0),
                )
                
                result.feedback.append({
                    "frame_index": idx,
                    "joint": worst_joint[0],
                    "error": round(worst_joint[1], 2),
                    "message": f"{worst_joint[0].replace('_', ' ').title()} angle differs by {worst_joint[1]:.1f}°",
                })
    
    return result

