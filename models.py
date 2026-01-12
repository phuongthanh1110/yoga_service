from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pydantic import Field


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


class FrameLandmarks(BaseModel):
    frame_index: int
    # For scoring we mainly need poseWorldLandmarks.
    # Keep other fields optional to avoid 422 when client sends compact payload.
    poseLandmarks: List[Landmark] = Field(default_factory=list)
    poseWorldLandmarks: List[Landmark] = Field(default_factory=list)
    faceLandmarks: List[Landmark] = Field(default_factory=list)
    leftHandLandmarks: List[Landmark] = Field(default_factory=list)
    rightHandLandmarks: List[Landmark] = Field(default_factory=list)
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


class PoseCompareRequest(BaseModel):
    """Request model for pose comparison."""
    # Accept compact payloads from Flutter (often only poseWorldLandmarks).
    # Use Dict[str, Any] to avoid 422 when optional fields are omitted.
    reference_frames: List[Dict[str, Any]]
    user_frames: List[Dict[str, Any]]
    reference_fps: float = 30.0
    user_fps: float = 30.0


class FeedbackItem(BaseModel):
    """Feedback for a specific frame."""
    frame_index: int
    joint: str
    error: float
    message: str


class PoseCompareResponse(BaseModel):
    """Response model for pose comparison."""
    overall_score: float
    angle_accuracy: float
    position_accuracy: float
    stability_score: float
    frame_scores: List[float]
    feedback: List[FeedbackItem]

