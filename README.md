# Multimodal Deepfake Detector

A full-stack deepfake detection project with a FastAPI backend, React/Vite frontend, and multimodal model pipeline for video, audio, and optional transcript evidence.

The app accepts a video upload, runs the analysis pipeline, and returns a final verdict with confidence, modality scores, suspicious frames, plots, and explanation text.

## Features

- Video-based deepfake scoring from sampled frames and face cues
- Audio-based scoring from waveform/spectrogram features
- Optional transcript/text evidence when available
- Weighted multimodal fusion for final prediction
- Suspicious frame preview and modality breakdown plots
- React frontend with upload, progress, and result views
- FastAPI backend with async job flow and sync analysis endpoint

## Project Structure

```text
multimodal-deepfake_-etector/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── utils/
│   │   └── extraction.py
│   │
│   ├── config.py
│   ├── inference_pipeline.py
│   ├── realtime.py
│   └── worker.py
│
├── data/
│   └── preprocess.py
│
├── frontend/
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── audio/
│   ├── fusion/
│   ├── text/
│   ├── video/
│   └── explainability.py
│
├── notebooks/
│
├── training/
│   ├── evaluate.py
│   ├── train_audio.py
│   └── train_video.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── README.md
```

## Requirements

- Python 3.11+
- Node.js 18+
- FFmpeg installed and available from the terminal
- Optional: CUDA-capable GPU for faster inference/training

## Backend Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run The Backend

```bash
uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend usually runs at:

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

By default, Vite proxies `/api` requests to the backend. To use a direct backend URL, set:

```bash
VITE_API_URL=http://127.0.0.1:8000
```

## API Endpoints

The FastAPI app is defined in `backend/api/main.py`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Check backend health. |
| `POST` | `/api/upload` | Upload a video and receive a job id. |
| `POST` | `/api/analyze/{job_id}` | Start analysis for an uploaded video. |
| `GET` | `/api/result/{job_id}` | Poll analysis status/result. |
| `POST` | `/api/analyze_sync` | Upload and analyze in one request. |
| `GET` | `/api/jobs` | List known jobs. |
| `DELETE` | `/api/job/{job_id}` | Delete a job. |

Typical result fields:

- `label`
- `fake_probability`
- `confidence`
- `video_score`
- `audio_score`
- `text_score`
- `modality_details`
- `explanation`
- `suspicious_frames`
- `audio_plot`
- `breakdown_plot`
- `processing_time_s`

## Training

Train the video model:

```bash
python -m training.train_video ^
  --data_dir data/processed/video ^
  --save_dir data/models ^
  --backbone efficientnet_b4 ^
  --epochs 30 ^
  --batch_size 32 ^
  --lr 1e-4 ^
  --device cuda
```

Train the audio model:

```bash
python -m training.train_audio ^
  --data_dir data/processed/audio ^
  --save_dir data/models ^
  --epochs 40 ^
  --batch_size 64 ^
  --device cuda
```

Evaluate saved models:

```bash
python -m training.evaluate ^
  --data_dir data/processed/video/test ^
  --video_model data/models/video_model.pth ^
  --audio_model data/models/audio_model.pth ^
  --output_dir eval_results
```

For macOS/Linux, replace `^` line continuations with `\`.

## Model Artifacts

Large model checkpoints are not GitHub-friendly. Do not commit files such as:

- `*.pth`
- `*.pt`
- `*.onnx`
- local datasets under `data/raw/` or `data/processed/`

For a public repo, upload checkpoints to a release, cloud drive, or model registry and add the download link here.

## Docker

Docker support files are included:

- `Dockerfile`
- `docker-compose.yml`
- `nginx.conf`

Local FastAPI + Vite development is recommended while iterating.

## Notes

- If a video has no usable audio, the pipeline can skip audio/transcript analysis instead of failing the whole request.
- The text branch is optional and should fail gracefully when external model downloads are unavailable.

## Dataset

This project uses the **FakeAVCeleb** dataset for training and evaluation of multimodal deepfake detection models.

Dataset Source:
https://github.com/DASH-Lab/FakeAVCeleb