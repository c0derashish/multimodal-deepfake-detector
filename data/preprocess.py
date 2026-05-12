"""
data/preprocess.py — Dataset preparation for training.

Supports:
  - FaceForensics++ (FF++)
  - Celeb-DF v2
  - ASVspoof 2019/2021 (audio)
  - FakeAVCeleb (multimodal)

Usage:
    # Prepare video dataset (extract face crops)
    python -m data.preprocess video \
        --input_dir /path/to/ff++ \
        --output_dir /path/to/output \
        --dataset ff++

    # Prepare audio dataset
    python -m data.preprocess audio \
        --input_dir /path/to/asvspoof \
        --output_dir /path/to/audio_out \
        --dataset asvspoof
"""

import argparse
import sys
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings


# ─── Face Crop Extractor ──────────────────────────────────────────────────────

class DatasetPreprocessor:
    """Extract face crops from videos and organise into real/fake directories."""

    def __init__(self, output_dir: Path, image_size: int = 224):
        self.output_dir  = output_dir
        self.image_size  = image_size
        self.real_dir    = output_dir / "real"
        self.fake_dir    = output_dir / "fake"
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)

        # Lazy import MTCNN to avoid heavy loading at module level
        from facenet_pytorch import MTCNN
        self.detector = MTCNN(
            image_size=image_size,
            margin=20,
            min_face_size=40,
            keep_all=False,
            device="cpu",
        )

    def process_video(
        self,
        video_path: Path,
        label: str,                     # "real" | "fake"
        max_frames: int = 20,
        sample_rate: int = 10,
    ) -> int:
        """
        Extract face crops from a single video.
        Returns number of saved crops.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Cannot open: {video_path}")
            return 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        save_dir = self.real_dir if label == "real" else self.fake_dir
        saved = 0
        frame_idx = 0

        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                face = self.detector(rgb)
                if face is not None:
                    # face is (C, H, W) tensor in [0,1]
                    face_np = (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    out_path = save_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out_path), face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
            except Exception:
                pass
            frame_idx += 1

        cap.release()
        return saved

    def process_ff_plus_plus(self, input_dir: Path, compression: str = "c23"):
        """
        FaceForensics++ structure:
            input_dir/
                original_sequences/actors/raw/  (real videos)
                manipulated_sequences/
                    Deepfakes/{compression}/videos/
                    Face2Face/{compression}/videos/
                    FaceSwap/{compression}/videos/
                    NeuralTextures/{compression}/videos/
        """
        logger.info("Processing FaceForensics++ dataset")

        real_videos = list((input_dir / "original_sequences" / "actors" / "raw").glob("**/*.mp4"))
        fake_dirs = [
            input_dir / "manipulated_sequences" / m / compression / "videos"
            for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        ]
        fake_videos = []
        for d in fake_dirs:
            if d.exists():
                fake_videos.extend(d.glob("**/*.mp4"))

        self._batch_process(real_videos, "real")
        self._batch_process(fake_videos, "fake")

    def process_celeb_df(self, input_dir: Path):
        """
        Celeb-DF v2 structure:
            input_dir/
                Celeb-real/   (real videos)
                Celeb-synthesis/ (fake videos)
                YouTube-real/ (real YouTube clips)
        """
        logger.info("Processing Celeb-DF v2 dataset")
        real_dirs = ["Celeb-real", "YouTube-real"]
        fake_dirs = ["Celeb-synthesis"]

        real_videos = []
        for d in real_dirs:
            real_videos.extend((input_dir / d).glob("**/*.mp4"))

        fake_videos = []
        for d in fake_dirs:
            fake_videos.extend((input_dir / d).glob("**/*.mp4"))

        self._batch_process(real_videos, "real")
        self._batch_process(fake_videos, "fake")

    def _batch_process(self, videos: List[Path], label: str, workers: int = 4):
        logger.info(f"Processing {len(videos)} {label} videos with {workers} workers")
        total_crops = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.process_video, v, label): v for v in videos}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  {label}"):
                try:
                    total_crops += fut.result()
                except Exception as e:
                    logger.warning(f"Failed: {futures[fut].name}: {e}")

        logger.info(f"Extracted {total_crops} {label} face crops")


# ─── Audio Dataset Preparation ────────────────────────────────────────────────

class AudioPreprocessor:
    """Prepare audio datasets (ASVspoof / FakeAVCeleb)."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        (output_dir / "real").mkdir(parents=True, exist_ok=True)
        (output_dir / "fake").mkdir(parents=True, exist_ok=True)

    def process_asvspoof(self, input_dir: Path, subset: str = "LA"):
        """
        ASVspoof 2019/2021 structure:
            input_dir/
                ASVspoof2019_{subset}_train/
                    flac/    (audio files)
                    ASVspoof2019_{subset}_train.trl.txt  (labels)
        """
        import soundfile as sf

        label_file = input_dir / f"ASVspoof2019_{subset}_train" / f"ASVspoof2019_{subset}_train.trl.txt"
        audio_dir  = input_dir / f"ASVspoof2019_{subset}_train" / "flac"

        if not label_file.exists():
            logger.error(f"Label file not found: {label_file}")
            return

        logger.info(f"Processing ASVspoof ({subset}) from {input_dir}")
        processed = 0

        with open(label_file) as f:
            for line in tqdm(f, desc="ASVspoof"):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                utt_id = parts[1]
                is_real = parts[4] == "bonafide"
                label = "real" if is_real else "fake"

                src = audio_dir / f"{utt_id}.flac"
                if not src.exists():
                    continue

                dst = self.output_dir / label / f"{utt_id}.wav"
                # Convert flac → wav using soundfile
                try:
                    data, sr = sf.read(str(src))
                    sf.write(str(dst), data, sr)
                    processed += 1
                except Exception as e:
                    logger.debug(f"Skip {utt_id}: {e}")

        logger.info(f"Processed {processed} ASVspoof audio files")

    def process_fakeavceleb(self, input_dir: Path):
        """
        FakeAVCeleb structure:
            input_dir/
                RealVideo-RealAudio/ → real
                FakeVideo-RealAudio/ → fake audio too (GAN voices mixed in)
                RealVideo-FakeAudio/ → fake audio
                FakeVideo-FakeAudio/ → fake audio
        """
        import subprocess

        real_audio_dirs = ["real"]
        fake_audio_dirs = ["fake"]

        for d in real_audio_dirs:
            self._extract_audio_from_videos(input_dir / d, "real")

        for d in fake_audio_dirs:
            self._extract_audio_from_videos(input_dir / d, "fake")

    def _extract_audio_from_videos(self, src_dir: Path, label: str):
        if not src_dir.exists():
            return
        videos = list(src_dir.glob("**/*.mp4")) + list(src_dir.glob("**/*.avi"))
        logger.info(f"Extracting audio from {len(videos)} {label} videos")

        import subprocess
        for video in tqdm(videos, desc=f"  {label} audio"):
            out_wav = self.output_dir / label / f"{video.stem}.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(video),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", str(out_wav),
            ]
            subprocess.run(cmd, capture_output=True)


# ─── Split Utilities ──────────────────────────────────────────────────────────

def create_train_val_test_split(
    data_dir: Path,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """
    Create train/val/test splits and save a JSON manifest.
    """
    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}

    for label in ["real", "fake"]:
        files = list((data_dir / label).glob("*"))
        random.shuffle(files)
        n = len(files)
        train_end = int(n * ratios[0])
        val_end   = train_end + int(n * ratios[1])

        for split, group in [
            ("train", files[:train_end]),
            ("val",   files[train_end:val_end]),
            ("test",  files[val_end:]),
        ]:
            splits[split].extend(
                [{"path": str(f), "label": label} for f in group]
            )

    manifest_path = data_dir / "splits.json"
    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)

    for split, items in splits.items():
        logger.info(f"  {split}: {len(items)} samples")

    logger.success(f"Manifest saved to {manifest_path}")
    return splits


# ─── Dataset Statistics ───────────────────────────────────────────────────────

def print_dataset_stats(data_dir: Path):
    """Print class distribution and basic statistics."""
    for label in ["real", "fake"]:
        d = data_dir / label
        if d.exists():
            files = list(d.glob("*"))
            logger.info(f"  {label}: {len(files)} files")

        # Compute a sample of file sizes
        sizes = [f.stat().st_size / 1024 for f in files[:100]]
        if sizes:
            logger.info(
                f"    Size: min={min(sizes):.1f}KB  "
                f"mean={sum(sizes)/len(sizes):.1f}KB  "
                f"max={max(sizes):.1f}KB"
            )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Deepfake dataset preprocessor")
    sub = p.add_subparsers(dest="mode")

    # Video mode
    vp = sub.add_parser("video", help="Extract face crops from videos")
    vp.add_argument("--input_dir",  required=True)
    vp.add_argument("--output_dir", required=True)
    vp.add_argument("--dataset", choices=["ff++", "celeb_df", "custom"], default="ff++")
    vp.add_argument("--compression", default="c23", help="FF++ compression level")
    vp.add_argument("--workers", type=int, default=4)

    # Audio mode
    ap = sub.add_parser("audio", help="Prepare audio datasets")
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset", choices=["asvspoof", "fakeavceleb", "custom"], default="asvspoof")

    # Split mode
    sp = sub.add_parser("split", help="Create train/val/test manifest")
    sp.add_argument("--data_dir", required=True)
    sp.add_argument("--train", type=float, default=0.70)
    sp.add_argument("--val",   type=float, default=0.15)
    sp.add_argument("--test",  type=float, default=0.15)

    args = p.parse_args()

    if args.mode == "video":
        proc = DatasetPreprocessor(Path(args.output_dir))
        if args.dataset == "ff++":
            proc.process_ff_plus_plus(Path(args.input_dir), args.compression)
        elif args.dataset == "celeb_df":
            proc.process_celeb_df(Path(args.input_dir))
        else:
            # Custom: expects input_dir/{real,fake}/**/*.mp4
            for label in ["real", "fake"]:
                videos = list((Path(args.input_dir) / label).glob("**/*.mp4"))
                proc._batch_process(videos, label, args.workers)

        print_dataset_stats(Path(args.output_dir))
        create_train_val_test_split(Path(args.output_dir))

    elif args.mode == "audio":
        proc = AudioPreprocessor(Path(args.output_dir))
        if args.dataset == "asvspoof":
            proc.process_asvspoof(Path(args.input_dir))
        elif args.dataset == "fakeavceleb":
            proc.process_fakeavceleb(Path(args.input_dir))

        print_dataset_stats(Path(args.output_dir))
        create_train_val_test_split(Path(args.output_dir))

    elif args.mode == "split":
        create_train_val_test_split(
            Path(args.data_dir),
            ratios=(args.train, args.val, args.test),
        )

    else:
        p.print_help()


if __name__ == "__main__":
    main()