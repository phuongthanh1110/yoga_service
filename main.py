from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import PoseExtractionResponse
from pose_processor import extract_world_landmarks

app = FastAPI(title="MediaPipe Pose Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename or "").suffix
    ) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

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

