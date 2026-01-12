"""Align two pose sequences with different timings."""

from typing import List, Dict, Tuple, Optional


def linear_interpolate_pose(
    pose1: dict,
    pose2: dict,
    t: float,  # 0.0 = pose1, 1.0 = pose2
) -> dict:
    """
    Interpolate between two poses.
    
    Args:
        pose1: First pose frame
        pose2: Second pose frame
        t: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated pose frame
    """
    if t <= 0.0:
        return pose1.copy()
    if t >= 1.0:
        return pose2.copy()
    
    result = pose1.copy()
    
    # Interpolate landmarks
    for landmark_key in ["poseWorldLandmarks", "poseLandmarks"]:
        if landmark_key in pose1 and landmark_key in pose2:
            landmarks1 = pose1[landmark_key]
            landmarks2 = pose2[landmark_key]
            
            if not landmarks1 or not landmarks2:
                continue
            
            min_len = min(len(landmarks1), len(landmarks2))
            interpolated = []
            
            for i in range(min_len):
                lm1 = landmarks1[i]
                lm2 = landmarks2[i]
                
                if not isinstance(lm1, dict) or not isinstance(lm2, dict):
                    continue
                
                interpolated_lm = {
                    "x": lm1.get("x", 0) * (1 - t) + lm2.get("x", 0) * t,
                    "y": lm1.get("y", 0) * (1 - t) + lm2.get("y", 0) * t,
                    "z": lm1.get("z", 0) * (1 - t) + lm2.get("z", 0) * t,
                }
                
                if "visibility" in lm1:
                    interpolated_lm["visibility"] = (
                        lm1.get("visibility", 0) * (1 - t) + lm2.get("visibility", 0) * t
                    )
                
                interpolated.append(interpolated_lm)
            
            result[landmark_key] = interpolated
    
    return result


def align_pose_sequences(
    reference_frames: List[dict],
    user_frames: List[dict],
    reference_fps: float = 30.0,
    user_fps: float = 30.0,
) -> List[Tuple[dict, dict]]:
    """
    Align two pose sequences by time.
    
    Maps each reference frame to corresponding user frame(s).
    Uses linear interpolation if needed.
    
    Args:
        reference_frames: Reference pose frames
        user_frames: User pose frames
        reference_fps: FPS of reference sequence
        user_fps: FPS of user sequence
    
    Returns:
        List of (reference_frame, aligned_user_frame) tuples
    """
    if not reference_frames or not user_frames:
        return []
    
    aligned_pairs = []
    
    # Calculate time per frame
    ref_time_per_frame = 1.0 / reference_fps if reference_fps > 0 else 1.0 / 30.0
    user_time_per_frame = 1.0 / user_fps if user_fps > 0 else 1.0 / 30.0
    
    # Total duration
    ref_duration = len(reference_frames) * ref_time_per_frame
    user_duration = len(user_frames) * user_time_per_frame
    
    # Scale user time to match reference duration
    time_scale = ref_duration / user_duration if user_duration > 0 else 1.0
    
    for ref_idx, ref_frame in enumerate(reference_frames):
        ref_time = ref_idx * ref_time_per_frame
        
        # Map to user time
        user_time = ref_time / time_scale
        
        # Find corresponding user frame index
        user_frame_idx = user_time / user_time_per_frame
        
        # Get user frames for interpolation
        user_idx_low = int(user_frame_idx)
        user_idx_high = min(user_idx_low + 1, len(user_frames) - 1)
        
        if user_idx_low < 0:
            user_idx_low = 0
        
        user_frame_low = user_frames[user_idx_low]
        user_frame_high = user_frames[user_idx_high] if user_idx_high > user_idx_low else user_frame_low
        
        # Interpolate if needed
        if user_idx_high > user_idx_low:
            t = user_frame_idx - user_idx_low
            aligned_user_frame = linear_interpolate_pose(user_frame_low, user_frame_high, t)
        else:
            aligned_user_frame = user_frame_low.copy()
        
        aligned_pairs.append((ref_frame, aligned_user_frame))
    
    return aligned_pairs

