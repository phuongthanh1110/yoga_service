# MediaPipe Pose Backend (FastAPI)

Extract MediaPipe Pose **world landmarks** from uploaded videos.

## Quick start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## API

- `POST /pose/extract?stride=1`  
  Multipart form field `file` (video). Returns JSON:
  ```json
  {
    "frames": [
      {
        "frame_index": 0,
        "landmarks": [
          {"x": 0.1, "y": -0.2, "z": -0.05, "visibility": 0.9},
          ...
        ]
      }
    ],
    "frame_count": 300,
    "fps": 30.0,
    "width": 1280,
    "height": 720
  }
  ```

- `GET /health` → `{"status": "ok"}`

### Example curl

```bash
curl -X POST "http://localhost:8000/pose/extract?stride=2" \
  -F "file=@/path/to/video.mp4" \
  | jq .
```

## Notes
- Uses MediaPipe Pose world landmarks (3D). `stride` can downsample frames.
- Requires Python + system deps for OpenCV; use `opencv-python-headless` to avoid GUI libs.

