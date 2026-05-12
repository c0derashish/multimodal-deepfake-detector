"""
models/video/video_model.py — Video deepfake detector.

Architecture:
  1. MTCNN  → detect & align faces per frame
  2. EfficientNet-B4 backbone → frame-level feature embeddings
  3. Temporal aggregation (mean + max pooling)
  4. MLP head → binary classification (real / fake)

Detects:
  - Blending artifacts at face boundary
  - Texture anomalies (GAN fingerprints)
  - Eye-blink / facial-motion inconsistencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

import timm
from facenet_pytorch import MTCNN


# ─── Face Detection & Alignment ───────────────────────────────────────────────

class FaceExtractor:
    """Detect faces in a frame using MTCNN and return aligned crops."""

    def __init__(self, device: str = "cpu", image_size: int = 224):
        self.device = device
        self.image_size = image_size
        self.detector = MTCNN(
            image_size=image_size,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.9],
            factor=0.709,
            keep_all=False,
            device=device,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the largest face from a BGR frame.
        Returns aligned RGB numpy array (H, W, 3) or None.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            face_tensor = self.detector(frame_rgb)
            if face_tensor is None:
                return None
            # facenet-pytorch returns (C, H, W) float tensor [0,1]
            face_np = face_tensor.permute(1, 2, 0).numpy()
            face_np = (face_np * 255).clip(0, 255).astype(np.uint8)
            return face_np
        except Exception as e:
            logger.debug(f"MTCNN error: {e}")
            return None


# ─── Model Architecture ───────────────────────────────────────────────────────

class VideoDeepfakeDetector(nn.Module):
    """
    Frame-level deepfake classifier with temporal aggregation.

    Input:  (B, T, C, H, W)  — batch of T frames per sample
    Output: (B, 1)            — logit (sigmoid → fake probability)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # remove classifier head
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        # Optionally freeze early layers
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "blocks.0" in name or "blocks.1" in name:
                    param.requires_grad = False

        # ── Temporal attention ────────────────────────────────────────────────
        # Lightweight self-attention across frames
        self.frame_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),          # concat mean+max
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

        # Artifact detection branch (texture anomaly auxiliary loss)
        self.artifact_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,           # (B, T, C, H, W)
        return_frame_scores: bool = False,
    ) -> dict:
        B, T, C, H, W = x.shape

        # Reshape to process all frames together
        frames = x.view(B * T, C, H, W)            # (B*T, C, H, W)
        frame_feats = self.backbone(frames)          # (B*T, D)
        D = frame_feats.shape[-1]
        frame_feats = frame_feats.view(B, T, D)     # (B, T, D)

        # Temporal self-attention
        attn_out, attn_weights = self.frame_attn(
            frame_feats, frame_feats, frame_feats
        )                                           # (B, T, D)

        # Aggregate: mean + max → 2D
        mean_feat = attn_out.mean(dim=1)            # (B, D)
        max_feat  = attn_out.max(dim=1).values      # (B, D)
        agg = torch.cat([mean_feat, max_feat], dim=-1)   # (B, 2D)

        # Frame-level artifact scores (auxiliary)
        artifact_logits = self.artifact_head(frame_feats.view(B * T, D))
        artifact_logits = artifact_logits.view(B, T)    # (B, T)

        # Main classification
        logits = self.head(agg).squeeze(-1)             # (B,)

        out = {
            "logit": logits,
            "prob":  torch.sigmoid(logits),
            "artifact_logits": artifact_logits,
        }
        if return_frame_scores:
            out["frame_probs"] = torch.sigmoid(artifact_logits)  # (B, T)
            out["attn_weights"] = attn_weights                    # (B, T, T)
        return out


# ─── Inference Helper ─────────────────────────────────────────────────────────

class VideoInference:
    """High-level wrapper for video-level deepfake prediction."""

    # ImageNet-style normalisation (applied to face crops)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, model_path: Optional[Path], device: str = "cpu"):
        self.device = device
        self.face_extractor = FaceExtractor(device=device)

        self.model = VideoDeepfakeDetector(pretrained=False)
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            logger.info(f"Video model loaded from {model_path}")
        else:
            logger.warning("No trained video weights — using random init (for demo)")
        self.model.to(device).eval()

    def preprocess_face(self, face_rgb: np.ndarray) -> torch.Tensor:
        """Normalise a (H,W,3) uint8 face → (3,H,W) float tensor."""
        import torchvision.transforms.functional as TF
        from PIL import Image
        img = Image.fromarray(face_rgb).resize((224, 224))
        t = TF.to_tensor(img)                               # [0,1]
        t = TF.normalize(t, self.MEAN, self.STD)
        return t

    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> dict:
        """
        Predict deepfake probability from a list of BGR frames.
        Returns dict with 'fake_probability', 'frame_scores', 'explanation'.
        """
        face_tensors: List[torch.Tensor] = []
        face_indices: List[int] = []

        for i, frame in enumerate(frames):
            face = self.face_extractor.extract(frame)
            if face is not None:
                face_tensors.append(self.preprocess_face(face))
                face_indices.append(i)

        if not face_tensors:
            logger.warning("No faces detected in any frame")
            return {
                "fake_probability": 0.5,
                "face_detected": False,
                "frame_scores": [],
                "explanation": ["No faces detected — cannot assess video modality"],
            }

        # Stack → (1, T, C, H, W)
        batch = torch.stack(face_tensors, dim=0).unsqueeze(0).to(self.device)

        out = self.model(batch, return_frame_scores=True)
        fake_prob = float(out["prob"][0].cpu())
        frame_probs = out["frame_probs"][0].cpu().tolist()

        explanation = self._build_explanation(fake_prob, frame_probs, face_indices)

        return {
            "fake_probability": round(fake_prob, 4),
            "face_detected": True,
            "n_faces_analyzed": len(face_tensors),
            "frame_scores": [
                {"frame_idx": idx, "score": round(s, 4)}
                for idx, s in zip(face_indices, frame_probs)
            ],
            "explanation": explanation,
        }

    def _build_explanation(
        self,
        fake_prob: float,
        frame_scores: List[float],
        frame_indices: List[int],
    ) -> List[str]:
        reasons = []
        if fake_prob > 0.8:
            reasons.append(f"Strong fake signal detected in video stream (score={fake_prob:.2f})")
        elif fake_prob > 0.5:
            reasons.append(f"Moderate fake signal in video (score={fake_prob:.2f})")
        else:
            reasons.append(f"Video appears authentic (score={fake_prob:.2f})")

        high_frames = [idx for idx, s in zip(frame_indices, frame_scores) if s > 0.6]
        if high_frames:
            reasons.append(f"Suspicious frames detected at indices: {high_frames[:5]}")

        variance = float(np.var(frame_scores)) if frame_scores else 0.0
        if variance > 0.05:
            reasons.append("High temporal inconsistency detected across frames")

        return reasons