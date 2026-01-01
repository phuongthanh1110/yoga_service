# MediaPipe Holistic Backend (FastAPI)

Extract MediaPipe Holistic **pose + hands + face landmarks** from uploaded
videos and expose them for 3D retargeting.

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
  Multipart form field `file` (video). Returns JSON (shortened):
  ```json
  {
    "metadata": {...},
    "poseLandmarks": [...],            // first-frame image coords
    "poseWorldLandmarks": [...],       // first-frame world coords
    "leftHandLandmarks": [...],
    "rightHandLandmarks": [...],
    "leftHandWorldLandmarks": [...],
    "rightHandWorldLandmarks": [...],
    "faceLandmarks": [...],
    "landmarkIndices": {...},          // pose index->name
    "handLandmarkIndices": {...},      // hand index->name
    "faceLandmarkIndices": {...},      // face index->name
    "frames": [
      {
        "frame_index": 0,
        "poseLandmarks": [...],
        "poseWorldLandmarks": [...],
        "leftHandLandmarks": [...],
        "rightHandLandmarks": [...],
        "leftHandWorldLandmarks": [...],
        "rightHandWorldLandmarks": [...],
        "faceLandmarks": [...],
        "segmentationMask": null
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

## Expose locally with ngrok

```bash
# Install once (macOS):
brew install ngrok/ngrok/ngrok

# (Optional) set your auth token
ngrok config add-authtoken <token>

# Start your API locally (from above)
uvicorn main:app --reload --port 8000

# In a new terminal, tunnel port 8000
ngrok http 8000
```

ngrok will print a public URL like `https://<id>.ngrok.io` that forwards to your local server. Use that URL for external clients while testing.

## Notes
- Uses MediaPipe Pose world landmarks (3D). `stride` can downsample frames.
- Requires Python + system deps for OpenCV; use `opencv-python-headless` to avoid GUI libs.

