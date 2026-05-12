"""
data/generate_dummy_data.py
Generates synthetic real/fake videos and audio for testing
the full pipeline without any real dataset.
"""

import cv2
import numpy as np
import soundfile as sf
from pathlib import Path
import random

def create_dummy_video(path: Path, label: str, n_frames: int = 60, fps: int = 25):
    """
    Create a synthetic video.
    Real = smooth face-like oval with natural motion
    Fake = same but with added blending artifacts
    """
    h, w = 224, 224
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Skin-colored background
        frame[:] = (180, 150, 120)

        # Face oval
        cx, cy = w // 2, h // 2
        # Slight natural motion
        offset_x = int(5 * np.sin(i * 0.1))
        offset_y = int(3 * np.cos(i * 0.15))
        cv2.ellipse(frame, (cx + offset_x, cy + offset_y),
                    (60, 80), 0, 0, 360, (210, 175, 140), -1)

        # Eyes
        eye_blink = 1 if (i % 30 != 0) else 0  # blink every 30 frames
        if eye_blink:
            cv2.ellipse(frame, (cx - 20 + offset_x, cy - 15 + offset_y),
                        (8, 5), 0, 0, 360, (50, 30, 20), -1)
            cv2.ellipse(frame, (cx + 20 + offset_x, cy - 15 + offset_y),
                        (8, 5), 0, 0, 360, (50, 30, 20), -1)

        # Mouth
        cv2.ellipse(frame, (cx + offset_x, cy + 25 + offset_y),
                    (15, 6), 0, 0, 180, (120, 60, 60), -1)

        if label == "fake":
            # Add blending artifacts (hard edge, color mismatch)
            artifact_y = cy + offset_y + 40
            cv2.rectangle(frame,
                          (cx - 55, artifact_y),
                          (cx + 55, artifact_y + 8),
                          (140, 200, 160), -1)
            # Add noise patch
            noise = np.random.randint(0, 60,
                    (20, 40, 3), dtype=np.uint8)
            frame[cy-10:cy+10, cx+30:cx+70] = noise
        else:
            # Real: just slight gaussian noise
            noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise,
                           0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()


def create_dummy_audio(path: Path, label: str,
                       duration: float = 3.0, sr: int = 16000):
    """
    Real = natural speech-like signal (sum of harmonics)
    Fake = synthetic TTS-like signal (perfect harmonics, no noise)
    """
    t = np.linspace(0, duration, int(sr * duration))

    if label == "real":
        # Natural speech: multiple harmonics + noise + pitch variation
        pitch = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)  # varying pitch
        signal = (
            0.4 * np.sin(2 * np.pi * pitch * t) +
            0.2 * np.sin(2 * np.pi * pitch * 2 * t) +
            0.1 * np.sin(2 * np.pi * pitch * 3 * t) +
            0.05 * np.random.randn(len(t))  # natural noise
        )
    else:
        # Fake: perfect pitch, no variation (TTS artifact)
        pitch = 120.0  # perfectly constant
        signal = (
            0.5 * np.sin(2 * np.pi * pitch * t) +
            0.3 * np.sin(2 * np.pi * pitch * 2 * t) +
            0.1 * np.sin(2 * np.pi * pitch * 3 * t)
            # no noise — too perfect
        )

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    sf.write(str(path), signal.astype(np.float32), sr)


def generate_dataset(
    base_dir: str = "data",
    n_train: int = 50,   # videos per class for training
    n_val:   int = 10,
    n_test:  int = 10,
):
    base = Path(base_dir)

    # Video dataset
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (base / "processed" / "video" / split / label).mkdir(
                parents=True, exist_ok=True)

    # Audio dataset
    for label in ["real", "fake"]:
        (base / "processed" / "audio" / label).mkdir(
            parents=True, exist_ok=True)

    # Raw video (for pipeline testing)
    for label in ["real", "fake"]:
        (base / "raw" / label).mkdir(parents=True, exist_ok=True)

    counts = {"train": n_train, "val": n_val, "test": n_test}

    print("Generating synthetic videos...")
    total = sum(counts.values()) * 2
    done = 0

    for split, count in counts.items():
        for label in ["real", "fake"]:
            for i in range(count):
                # Video file
                vid_path = (base / "processed" / "video" /
                            split / label / f"{label}_{i:04d}.mp4")
                create_dummy_video(vid_path, label,
                                   n_frames=random.randint(40, 90))

                # Also save to raw/ for full pipeline testing
                if split == "train" and i < 5:
                    raw_path = (base / "raw" / label /
                                f"{label}_{i:04d}.mp4")
                    create_dummy_video(raw_path, label)

                done += 1
                print(f"  [{done}/{total}] {split}/{label}/{label}_{i:04d}.mp4")

    print("\nGenerating synthetic audio...")
    for label in ["real", "fake"]:
        for i in range(n_train + n_val + n_test):
            aud_path = (base / "processed" / "audio" /
                        label / f"{label}_{i:04d}.wav")
            create_dummy_audio(aud_path, label,
                               duration=random.uniform(2.0, 5.0))

    print(f"""
Done! Dataset created at {base}/
  Video: {n_train+n_val+n_test} real + {n_train+n_val+n_test} fake
  Audio: {n_train+n_val+n_test} real + {n_train+n_val+n_test} fake

Next steps:
  Train video model:
    python -m training.train_video --data_dir data/processed/video/train --save_dir data/models

  Train audio model:
    python -m training.train_audio --data_dir data/processed/audio --save_dir data/models

  Or just start the API:
    uvicorn backend.api.main:app --reload --port 8000
""")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="data")
    p.add_argument("--n_train",  type=int, default=50)
    p.add_argument("--n_val",    type=int, default=10)
    p.add_argument("--n_test",   type=int, default=10)
    args = p.parse_args()
    generate_dataset(args.base_dir, args.n_train, args.n_val, args.n_test)




# python -m training.train_video --data_dir data/processed/video --save_dir data/models --epochs 10 --batch_size 8 --device cpu

# python -m training.train_audio --data_dir data/processed/audio --save_dir data/models --epochs 5 --batch_size 16 --device cpu

# python -m data.preprocess video --input_dir data/raw --output_dir data/processed/video --dataset custom




# python -m data.preprocess video --input_dir data/raw/video --output_dir data/processed/video --dataset custom

# python -m data.preprocess audio --input_dir data/raw/audio --output_dir data/processed/audio --dataset fakeavceleb