from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


class FrameLandmarks(BaseModel):
    frame_index: int
    poseLandmarks: List[Landmark]
    poseWorldLandmarks: List[Landmark]
    faceLandmarks: List[Landmark]
    leftHandLandmarks: List[Landmark]
    rightHandLandmarks: List[Landmark]
    segmentationMask: Optional[str] = None  # always null in current export


class Metadata(BaseModel):
    timestamp: str
    modelUrl: str
    exportVersion: str
    source: str


class PoseExtractionResponse(BaseModel):
    metadata: Metadata
    poseLandmarks: List[Landmark]  # first frame convenience mirror
    poseWorldLandmarks: List[Landmark]  # first frame convenience mirror
    faceLandmarks: List[Landmark]  # first frame convenience mirror
    leftHandLandmarks: List[Landmark]  # first frame convenience mirror
    rightHandLandmarks: List[Landmark]  # first frame convenience mirror
    segmentationMask: Optional[str] = None
    landmarkIndices: Dict[int, str]
    frames: List[FrameLandmarks]
    frame_count: int
    fps: float
    width: int
    height: int


# ============================================================================
# Pose Comparison Models (AQA - Action Quality Assessment)
# ============================================================================

class TrainerPoseMetadata(BaseModel):
    """Metadata for a trainer's reference pose sequence."""
    id: str
    name: str
    description: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    duration_seconds: float
    created_at: str
    category: Optional[str] = None  # e.g., "standing", "seated", "balance"


class TrainerPoseData(BaseModel):
    """Stored trainer pose data (pre-calculated features)."""
    metadata: TrainerPoseMetadata
    frames: List[Dict[str, Any]]  # Raw landmark frames
    fps: float
    frame_count: int


class JointScoreDetail(BaseModel):
    """Score detail for a specific joint."""
    score: float = Field(..., ge=0, le=100)
    average_error_degrees: float
    weight: float


class FeedbackItem(BaseModel):
    """A single feedback item with correction guidance."""
    timestamp: float  # seconds into the video
    joint: str
    message: str
    error_degrees: float
    severity: str  # low, medium, high


class AlignmentInfo(BaseModel):
    """Information about DTW alignment."""
    trainer_frames: int
    user_frames: int
    aligned_pairs: int


class ComparisonMetadata(BaseModel):
    """Metadata about the comparison process."""
    trainer_fps: float
    user_fps: float
    joints_analyzed: int


class PoseComparisonResult(BaseModel):
    """Result of comparing user's pose to trainer's reference."""
    success: bool
    error: Optional[str] = None
    overall_score: float = Field(0.0, ge=0, le=100)
    dtw_distance: Optional[float] = None
    average_angle_error: Optional[float] = None
    joint_scores: Dict[str, JointScoreDetail] = {}
    feedback: List[FeedbackItem] = []
    alignment_info: Optional[AlignmentInfo] = None
    metadata: Optional[ComparisonMetadata] = None


class TrainerUploadResponse(BaseModel):
    """Response after uploading a trainer reference video."""
    success: bool
    trainer_id: str
    message: str
    metadata: TrainerPoseMetadata


class TrainerListItem(BaseModel):
    """Item in trainer pose list."""
    id: str
    name: str
    description: Optional[str] = None
    difficulty: str
    duration_seconds: float
    category: Optional[str] = None


class TrainerListResponse(BaseModel):
    """Response with list of available trainer poses."""
    trainers: List[TrainerListItem]
    total: int

