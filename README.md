# MediaPipe Pose Backend (FastAPI)

Extract MediaPipe Pose **world landmarks** from uploaded videos + **Pose Comparison (AQA)**.

## Quick start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Pose Extraction

- `POST /pose/extract?stride=1`  
  Extract pose landmarks from a video.
  
  **Parameters:**
  - `file`: Video file (multipart form)
  - `stride`: Frame sampling rate (default: 1)
  
  **Response:**
  ```json
  {
    "frames": [
      {
        "frame_index": 0,
        "poseWorldLandmarks": [
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

- `POST /pose/extract/stream?stride=1`  
  Same as above but with SSE progress updates.

### Trainer Management (Reference Poses)

- `POST /trainer/upload`  
  Upload a trainer's reference yoga pose video.
  
  **Parameters (form data):**
  - `file`: Video file
  - `name`: Pose name (e.g., "Warrior II")
  - `description`: Optional description
  - `difficulty`: easy/medium/hard (default: medium)
  - `category`: Optional category (e.g., "standing", "balance")
  - `stride`: Frame sampling rate (default: 1)
  
  **Response:**
  ```json
  {
    "success": true,
    "trainer_id": "abc123",
    "message": "Trainer pose 'Warrior II' uploaded successfully",
    "metadata": {
      "id": "abc123",
      "name": "Warrior II",
      "difficulty": "medium",
      "duration_seconds": 15.5,
      "created_at": "2025-01-07T..."
    }
  }
  ```

- `GET /trainer/list?category=&difficulty=`  
  List all available trainer poses with optional filtering.

- `GET /trainer/{trainer_id}`  
  Get detailed trainer pose data.

- `DELETE /trainer/{trainer_id}`  
  Delete a trainer pose.

### Pose Comparison (AQA - Action Quality Assessment)

- `POST /compare/{trainer_id}?stride=1`  
  Compare user's video against a trainer's reference.
  
  **Parameters:**
  - `file`: User's video file
  - `trainer_id`: ID of trainer to compare against
  - `stride`: Frame sampling rate
  
  **Response:**
  ```json
  {
    "success": true,
    "overall_score": 85.5,
    "dtw_distance": 123.45,
    "average_angle_error": 12.3,
    "joint_scores": {
      "left_knee": {"score": 92.0, "average_error_degrees": 8.5, "weight": 1.5},
      "right_elbow": {"score": 78.0, "average_error_degrees": 18.2, "weight": 1.0}
    },
    "feedback": [
      {
        "timestamp": 3.5,
        "joint": "left_knee",
        "message": "Bend your left knee more deeply",
        "error_degrees": 25.0,
        "severity": "medium"
      }
    ],
    "alignment_info": {
      "trainer_frames": 150,
      "user_frames": 120,
      "aligned_pairs": 150
    }
  }
  ```

- `POST /compare/json/{trainer_id}`  
  Compare pre-extracted pose JSON against trainer (saves bandwidth).
  
  **Request body:**
  ```json
  {
    "frames": [...],
    "fps": 30.0
  }
  ```

- `GET /health` → `{"status": "ok"}`

### Example curl commands

```bash
# Extract poses from video
curl -X POST "http://localhost:8000/pose/extract?stride=2" \
  -F "file=@/path/to/video.mp4" | jq .

# Upload trainer pose
curl -X POST "http://localhost:8000/trainer/upload" \
  -F "file=@/path/to/trainer_video.mp4" \
  -F "name=Warrior II" \
  -F "difficulty=medium" | jq .

# List trainers
curl "http://localhost:8000/trainer/list" | jq .

# Compare user video against trainer
curl -X POST "http://localhost:8000/compare/abc123?stride=2" \
  -F "file=@/path/to/user_video.mp4" | jq .
```

## How Pose Comparison Works

The comparison uses **Action Quality Assessment (AQA)** techniques:

1. **Scale Invariance**: Compares joint angles instead of raw coordinates
   - Works regardless of user's body size
   - A 90° elbow is 90° whether you're tall or short

2. **Temporal Alignment (DTW)**: Uses Dynamic Time Warping
   - Handles speed differences between user and trainer
   - Automatically syncs timelines for fair comparison

3. **Weighted Scoring**: Different joints have different importance
   - Knees: 1.5x weight (critical for yoga stances)
   - Hips: 1.3x weight (core posture)
   - Elbows: 1.0x weight (arm positions)

## Expose locally with ngrok

```bash
# Install once (macOS):
brew install ngrok/ngrok/ngrok

# Start API
uvicorn main:app --reload --port 8000

# In new terminal, tunnel
ngrok http 8000
```

## Notes
- Trainer poses are stored in `backend/trainer_poses/` as JSON files
- Uses MediaPipe Pose world landmarks (3D)
- `stride` can downsample frames for faster processing
- Requires Python + OpenCV; uses `opencv-python-headless` to avoid GUI libs

