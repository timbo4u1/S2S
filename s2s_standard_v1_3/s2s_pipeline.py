"""
s2s_pipeline.py — Unified S2S Pipeline

Single entry point for the full S2S chain.
Wraps all 5 layers into one clean API.

Install:
    pip install s2s-certify
    pip install "s2s-certify[ml]"          # for Layer 4a/4b
    pip install "s2s-certify[research]"    # for scipy, wfdb

Optional extras (install separately):
    pip install git+https://github.com/openai/CLIP.git  # for Layer 5 video
    pip install sentence-transformers                    # for Layer 3/4c/5

Quick start:
    from s2s_standard_v1_3 import S2SPipeline

    pipe = S2SPipeline(segment="forearm")

    # Option 1 — IMU only
    result = pipe.certify(imu_raw={"timestamps_ns": ts, "accel": acc, "gyro": gyro})

    # Option 2 — IMU + text intent query
    result = pipe.certify(imu_raw={...}, instruction="pick up the cup")

    # Option 3 — Full chain: IMU + text + video frame
    result = pipe.certify(imu_raw={...}, instruction="pick up the cup",
                          video_frame=jpeg_bytes)

    print(result["tier"])          # GOLD / SILVER / BRONZE / REJECTED
    print(result["score"])         # 0-100
    print(result["source_type"])   # HIL_BIOLOGICAL
    print(result["intent"])        # "pick object" (0.887 similarity)
    print(result["next_motion"])   # 8-dim next window prediction (Layer 4a)
    print(result["clip_sim"])      # CLIP scene-instruction similarity (Layer 5)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

# Optional imports — degrade gracefully
try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except ImportError:
    _ST = False

try:
    import clip as _clip_lib
    _CLIP = True
except ImportError:
    _CLIP = False


class S2SPipeline:
    """
    Unified S2S pipeline — all 5 layers in one object.

    Parameters
    ----------
    segment : str
        Body segment. One of: forearm, upper_arm, hand, finger, head, walking
    experiments_dir : str
        Path to experiments/ directory containing trained models.
        Default: ~/S2S/experiments/
    device : str
        PyTorch device. Default: cpu
    verbose : bool
        Print progress messages. Default: False
    """

    def __init__(self,
                 segment: str = "forearm",
                 experiments_dir: Optional[str] = None,
                 device: str = "cpu",
                 verbose: bool = False):

        self.segment  = segment
        self.device   = device
        self.verbose  = verbose

        if experiments_dir is None:
            experiments_dir = os.path.expanduser("~/S2S/experiments")
        self.exp_dir = Path(experiments_dir)

        # Layer 1 — always available
        self.engine = PhysicsEngine()

        # Layer 4a model
        self._layer4a = None
        self._layer4a_mean = None
        self._layer4a_std  = None

        # Layer 4c model
        self._layer4c       = None
        self._layer4c_mean  = None
        self._layer4c_std   = None
        self._layer4c_labels = None
        self._layer4c_embs  = None

        # Layer 5 CLIP
        self._clip_model    = None
        self._clip_preprocess = None

        # sentence-transformers
        self._st_model = None

        self._load_models()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[S2S] {msg}")

    def _load_models(self):
        """Load all available trained models silently."""

        # Layer 4a
        model4a_path = self.exp_dir / "layer4_model.pt"
        if _TORCH and model4a_path.exists():
            try:
                import torch.nn as nn
                ckpt = torch.load(str(model4a_path), map_location=self.device)

                class ActionSeqModel(nn.Module):
                    def __init__(self, d_in, hidden, d_out, n_heads=4, n_layers=2):
                        super().__init__()
                        self.input_proj = nn.Linear(d_in, hidden)
                        enc = nn.TransformerEncoderLayer(
                            d_model=hidden, nhead=n_heads,
                            dim_feedforward=hidden*4, batch_first=True)
                        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
                        self.head = nn.Sequential(
                            nn.Linear(hidden, hidden), nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden, d_out))
                    def forward(self, x):
                        x = x.float()
                        if x.dim() == 2: x = x.unsqueeze(1)
                        x = self.input_proj(x)
                        x = self.transformer(x)
                        return self.head(x.mean(dim=1))

                cfg = ckpt.get("config", {})
                model = ActionSeqModel(
                    cfg.get("input_dim", 13),
                    cfg.get("hidden", 128),
                    cfg.get("output_dim", 8),
                    cfg.get("n_heads", 4),
                    cfg.get("n_layers", 2),
                )
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._layer4a      = model
                stats = ckpt.get("stats", {})
                self._layer4a_mean = np.array(stats.get("X_mean", ckpt.get("feat_mean", [0]*13)), dtype=np.float32)
                self._layer4a_std  = np.array(stats.get("X_std",  ckpt.get("feat_std",  [1]*13)), dtype=np.float32)
                self._log("Layer 4a loaded")
            except Exception as e:
                self._log(f"Layer 4a not loaded: {e}")

        # Layer 4c
        model4c_path = self.exp_dir / "layer4c_model.pt"
        label_embs_path = self.exp_dir / "layer4c_label_embs.npy"
        labels_path     = self.exp_dir / "layer4c_labels.json"
        if _TORCH and model4c_path.exists():
            try:
                import torch.nn as nn, json
                ckpt = torch.load(str(model4c_path), map_location=self.device)
                cfg  = ckpt["config"]

                class MotionClassifier(nn.Module):
                    def __init__(self, d_in, hidden, n_labels, proj_dim=384):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(d_in, hidden), nn.LayerNorm(hidden),
                            nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                            nn.ReLU(), nn.Dropout(0.1),
                            nn.Linear(hidden, hidden//2), nn.ReLU(),
                        )
                        self.classifier  = nn.Linear(hidden//2, n_labels)
                        self.projector   = nn.Sequential(
                            nn.Linear(hidden//2, proj_dim), nn.LayerNorm(proj_dim))

                    def forward(self, x):
                        return self.classifier(self.encoder(x.float()))

                    def embed(self, x):
                        h = self.encoder(x.float())
                        e = self.projector(h)
                        return nn.functional.normalize(e, dim=-1)

                model = MotionClassifier(
                    cfg["input_dim"], cfg.get("hidden", 256), cfg["n_labels"])
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._layer4c      = model
                self._layer4c_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
                self._layer4c_std  = np.array(ckpt["feat_std"],  dtype=np.float32)
                self._layer4c_labels = ckpt["unique_labels"]

                if label_embs_path.exists():
                    embs = np.load(str(label_embs_path)).astype(np.float32)
                    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
                    self._layer4c_embs = embs / norms
                self._log("Layer 4c loaded")
            except Exception as e:
                self._log(f"Layer 4c not loaded: {e}")

        # sentence-transformers
        if _ST:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._log("sentence-transformers loaded")
            except Exception:
                pass

        # CLIP
        if _CLIP:
            try:
                self._clip_model, self._clip_preprocess = \
                    _clip_lib.load("ViT-B/32", device=self.device)
                self._clip_model.eval()
                self._log("CLIP loaded")
            except Exception as e:
                self._log(f"CLIP not loaded: {e}")

    # ------------------------------------------------------------------
    # Feature extraction (13-dim, same as Layer 4 training)
    # ------------------------------------------------------------------

    def _extract_features(self, accel, hz: float = 50.0):
        if not _NP:
            return None
        a = np.array(accel, dtype=np.float64)
        if a.ndim == 1: a = a.reshape(-1, 1)
        while a.shape[1] < 3:
            a = np.hstack([a, np.zeros((len(a), 1))])
        g = np.zeros_like(a)
        f = [
            float(np.sqrt(np.mean(a**2))),
            float(np.std(a)),
            float(np.max(np.abs(a))),
            float(np.sqrt(np.mean(g**2))),
            float(np.std(g)),
        ]
        if len(a) > 3:
            jerk = np.diff(a, axis=0) * hz
            f.append(float(np.sqrt(np.mean(jerk**2))))
            f.append(float(np.percentile(np.abs(jerk), 95)))
        else:
            f.extend([0.0, 0.0])
        for ax in range(3):
            fft   = np.abs(np.fft.rfft(a[:, ax]))
            freqs = np.fft.rfftfreq(len(a), 1/hz)
            f.append(float(freqs[np.argmax(fft)] if len(fft) > 0 else 0))
        c = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
        f.append(float(c) if not np.isnan(c) else 0.0)
        f.append(float(np.linalg.norm(np.mean(a, axis=0))))
        hist, _ = np.histogram(a.flatten(), bins=20)
        hist = hist / (hist.sum() + 1e-10)
        f.append(float(-np.sum(hist * np.log(hist + 1e-10))))
        return np.array(f, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def certify(self,
                imu_raw: Dict,
                segment: Optional[str] = None,
                instruction: Optional[str] = None,
                video_frame: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Run the full S2S chain on one sensor window.

        Parameters
        ----------
        imu_raw : dict
            Must contain: timestamps_ns (list), accel (list of [x,y,z]),
            optionally: gyro (list of [x,y,z])
        segment : str, optional
            Override the pipeline segment for this call.
        instruction : str, optional
            Text description of the intended motion. Enables Layer 3/4c intent.
        video_frame : bytes, optional
            JPEG image bytes of the scene. Enables Layer 5 CLIP similarity.

        Returns
        -------
        dict with keys:
            tier          : GOLD / SILVER / BRONZE / REJECTED
            score         : int 0-100
            source_type   : HIL_BIOLOGICAL
            laws_passed   : list
            laws_failed   : list
            intent        : str or None  (top intent label)
            intent_sim    : float or None
            next_motion   : list or None  (8-dim Layer 4a prediction)
            clip_sim      : float or None  (Layer 5 CLIP similarity)
        """
        seg = segment or self.segment

        # ── Layer 1: Physics certification ───────────────────────────
        cert = self.engine.certify(imu_raw, segment=seg)

        result: Dict[str, Any] = {
            "tier":         cert.get("tier", "REJECTED"),
            "score":        cert.get("physical_law_score", 0),
            "source_type":  cert.get("source_type", "HIL_BIOLOGICAL"),
            "laws_passed":  cert.get("laws_passed", []),
            "laws_failed":  cert.get("laws_failed", []),
            "segment":      seg,
            # Layer 4 placeholders
            "intent":       None,
            "intent_sim":   None,
            "next_motion":  None,
            # Layer 5 placeholders
            "clip_sim":     None,
        }

        if not _NP:
            return result

        accel = imu_raw.get("accel", [])
        if not accel:
            return result

        # Sample rate from timestamps
        ts = imu_raw.get("timestamps_ns", [])
        hz = 50.0
        if len(ts) > 1:
            dt = (ts[1] - ts[0]) * 1e-9
            hz = 1.0 / dt if dt > 0 else 50.0

        feats = self._extract_features(accel, hz)
        if feats is None:
            return result

        # ── Layer 4c: Intent recognition ─────────────────────────────
        if self._layer4c is not None and _TORCH:
            try:
                x = torch.tensor(
                    ((feats - self._layer4c_mean) / self._layer4c_std
                     ).reshape(1, -1))
                with torch.no_grad():
                    logits = self._layer4c(x)[0]
                    probs  = torch.softmax(logits, dim=0).numpy()
                top_idx = int(np.argmax(probs))
                result["intent"]     = self._layer4c_labels[top_idx]
                result["intent_sim"] = round(float(probs[top_idx]), 4)
            except Exception:
                pass

        # ── Layer 3: Text-based intent override ──────────────────────
        if instruction and self._st_model and self._layer4c_embs is not None:
            try:
                q_emb = self._st_model.encode([instruction])[0]
                q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
                sims  = np.dot(self._layer4c_embs, q_emb)
                top   = int(np.argmax(sims))
                result["intent"]     = self._layer4c_labels[top]
                result["intent_sim"] = round(float(sims[top]), 4)
            except Exception:
                pass

        # ── Layer 4a: Next motion prediction ─────────────────────────
        if self._layer4a is not None and _TORCH:
            try:
                x = torch.tensor(
                    ((feats - self._layer4a_mean) / self._layer4a_std
                     ).reshape(1, -1))
                with torch.no_grad():
                    pred = self._layer4a(x)[0].numpy().tolist()
                result["next_motion"] = [round(v, 4) for v in pred]
            except Exception:
                pass

        # ── Layer 5: CLIP scene understanding ────────────────────────
        if video_frame and self._clip_model and instruction and _TORCH and _CLIP:
            try:
                import io
                from PIL import Image
                img = self._clip_preprocess(
                    Image.open(io.BytesIO(video_frame))
                ).unsqueeze(0)
                text = _clip_lib.tokenize([instruction[:77]])
                with torch.no_grad():
                    img_emb  = self._clip_model.encode_image(img)
                    text_emb = self._clip_model.encode_text(text)
                    img_emb  /= img_emb.norm(dim=-1, keepdim=True)
                    text_emb /= text_emb.norm(dim=-1, keepdim=True)
                    result["clip_sim"] = round(
                        float((img_emb @ text_emb.T)[0][0]), 4)
            except Exception:
                pass

        return result

    def certify_batch(self,
                      windows: List[Dict],
                      segment: Optional[str] = None) -> List[Dict]:
        """Certify a list of IMU windows. Returns list of result dicts."""
        return [self.certify(w, segment=segment) for w in windows]

    def query_intent(self, instruction: str, top_k: int = 5) -> List[tuple]:
        """
        Find top-k intent labels matching a text query.

        Returns list of (label, similarity) tuples.
        Requires sentence-transformers.
        """
        if not self._st_model or self._layer4c_embs is None:
            return []
        q = self._st_model.encode([instruction])[0]
        q = q / (np.linalg.norm(q) + 1e-8)
        sims    = np.dot(self._layer4c_embs, q)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._layer4c_labels[i], round(float(sims[i]), 4))
                for i in top_idx]

    def __repr__(self):
        layers = ["Layer1(Physics)"]
        if self._layer4a:  layers.append("Layer4a(NextAction)")
        if self._layer4c:  layers.append("Layer4c(Intent)")
        if self._st_model: layers.append("Layer3(Retrieval)")
        if self._clip_model: layers.append("Layer5(CLIP)")
        return f"S2SPipeline(segment={self.segment}, {' + '.join(layers)})"
