from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = None


class FrameLandmarks(BaseModel):
    frame_index: int
    poseLandmarks: List[Landmark]
    poseWorldLandmarks: List[Landmark]
    leftHandLandmarks: List[Landmark] = []
    rightHandLandmarks: List[Landmark] = []
    leftHandWorldLandmarks: List[Landmark] = []
    rightHandWorldLandmarks: List[Landmark] = []
    faceLandmarks: List[Landmark] = []
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
    leftHandLandmarks: List[Landmark] = []
    rightHandLandmarks: List[Landmark] = []
    leftHandWorldLandmarks: List[Landmark] = []
    rightHandWorldLandmarks: List[Landmark] = []
    faceLandmarks: List[Landmark] = []
    segmentationMask: Optional[str] = None
    landmarkIndices: Dict[int, str]
    handLandmarkIndices: Dict[int, str]
    faceLandmarkIndices: Dict[int, str]
    frames: List[FrameLandmarks]
    frame_count: int
    fps: float
    width: int
    height: int

