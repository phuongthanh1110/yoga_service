from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse

from models import PoseExtractionResponse
from pose_processor import extract_world_landmarks
import asyncio

app = FastAPI(title="MediaPipe Holistic Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZIP Compression: Giảm 50-70% response size
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB


@app.post("/pose/extract", response_model=PoseExtractionResponse)
async def extract_pose(file: UploadFile = File(...), stride: int = 1):
    if file.content_type:
        is_video = file.content_type.startswith("video/")
        is_octet = file.content_type == "application/octet-stream"
        if not (is_video or is_octet):
            # Fallback: allow common video extensions even if content_type is odd/missing
            filename = (file.filename or "").lower()
            allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
            if not any(filename.endswith(ext) for ext in allowed_ext):
                raise HTTPException(
                    status_code=400, detail="Please upload a video file."
                )

    # Determine file extension safely, default to .mp4 if unknown
    # Use safe filename to avoid encoding issues with special characters
    suffix = ".mp4"  # Default
    if file.filename:
        try:
            # Try to extract extension from filename safely
            filename_lower = file.filename.lower()
            for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                if filename_lower.endswith(ext):
                    suffix = ext
                    break
        except (UnicodeDecodeError, ValueError, AttributeError):
            # If filename has encoding issues, use default
            pass

    # Create temp file with unique name to avoid encoding issues
    # Use UUID-based filename instead of original filename
    tmp_path = Path(tempfile.gettempdir()) / f"yoga_pose_{uuid.uuid4().hex}{suffix}"
    
    # Write file content in binary mode to avoid encoding issues
    try:
        content = await file.read()
        with open(tmp_path, 'wb') as tmp:
            tmp.write(content)
    except Exception as e:
        # Clean up on error
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    try:
        result = extract_world_landmarks(str(tmp_path), stride=stride)
    except Exception as exc:  # pragma: no cover - thin API layer
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return PoseExtractionResponse(**result)


@app.post("/pose/extract/stream")
async def extract_pose_stream(file: UploadFile = File(...), stride: int = 1):
    """
    SSE endpoint for video processing with progress updates.
    Returns Server-Sent Events stream with progress and final result.
    """
    # Validate file type (same as regular endpoint)
    if file.content_type:
        is_video = file.content_type.startswith("video/")
        is_octet = file.content_type == "application/octet-stream"
        if not (is_video or is_octet):
            filename = (file.filename or "").lower()
            allowed_ext = (".mp4", ".mov", ".avi", ".mkv", ".webm")
            if not any(filename.endswith(ext) for ext in allowed_ext):
                raise HTTPException(
                    status_code=400, detail="Please upload a video file."
                )

    # Determine file extension
    suffix = ".mp4"
    if file.filename:
        try:
            filename_lower = file.filename.lower()
            for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                if filename_lower.endswith(ext):
                    suffix = ext
                    break
        except (UnicodeDecodeError, ValueError, AttributeError):
            pass

    tmp_path = Path(tempfile.gettempdir()) / f"yoga_pose_{uuid.uuid4().hex}{suffix}"

    # Read file content BEFORE creating async generator (to avoid "I/O operation on closed file")
    try:
        content = await file.read()
        with open(tmp_path, 'wb') as tmp:
            tmp.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to save uploaded file: {str(e)}"
        )

    async def generate():
        try:

            # Process video with progress updates
            import cv2
            from pose_processor import _get_mediapipe, _landmarks_to_list, _get_pose_landmark, _validate_hand_landmarks
            from datetime import datetime, timezone

            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                yield f"data: {json.dumps({'type': 'error', 'error': 'Unable to open video'})}\n\n"
                return

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

            if frame_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Video has no frames'})}\n\n"
                cap.release()
                return

            mp = _get_mediapipe()
            holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.65,  # Higher tracking confidence for smoother tracking
            )

            frames: list[dict] = []
            PoseLandmark = _get_pose_landmark()
            landmark_indices = {i: name.name.lower() for i, name in enumerate(PoseLandmark)}
            processed_count = 0

            # Calculate expected number of frames to process (with stride)
            expected_frames = (frame_count + stride - 1) // stride if stride > 1 else frame_count

            try:
                idx = 0
                while True:
                    success, frame = cap.read()
                    if not success:
                        break

                    if stride > 1 and (idx % stride) != 0:
                        idx += 1
                        continue

                    # Process frame
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)

                    # Convert and validate hand landmarks
                    left_hand = _landmarks_to_list(results.left_hand_landmarks)
                    right_hand = _landmarks_to_list(results.right_hand_landmarks)
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

                    processed_count += 1
                    # Send progress update (every 5 frames for more frequent updates)
                    if processed_count % 5 == 0 or processed_count == 1:
                        progress = min(99.0, (processed_count / expected_frames) * 100)
                        yield f"data: {json.dumps({'type': 'progress', 'progress': progress, 'processed': processed_count, 'total': expected_frames})}\n\n"
                        # Small delay to allow other tasks
                        await asyncio.sleep(0.01)

                    idx += 1

            finally:
                cap.release()
                holistic.close()

            # Prepare final result
            first_frame = frames[0] if frames else {
                "poseLandmarks": [],
                "poseWorldLandmarks": [],
                "faceLandmarks": [],
                "leftHandLandmarks": [],
                "rightHandLandmarks": [],
            }
            now_iso = datetime.now(timezone.utc).isoformat()

            result = {
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

            # Send final result
            yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Cleanup temp file
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering for nginx
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# For local dev: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


def _persist_result(result: dict) -> None:
    """
    Save the extracted pose JSON locally under backend/exports/.
    This is a side-effect only; it does not change the API response.
    """
    exports_dir = Path(__file__).parent / "exports"
    exports_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = exports_dir / f"pose_export_{timestamp}.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        # Fail silently; API response should not be blocked by disk issues.
        pass

