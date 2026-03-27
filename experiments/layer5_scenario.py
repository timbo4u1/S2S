#!/usr/bin/env python3
"""
layer5_scenario.py — S2S Layer 5: Scenario Understanding

Given a DROID episode (robot arm motion + language instruction + optional video),
run the full S2S chain:
  1. Parse DROID TFRecord → cartesian action sequence + language instruction
  2. Certify motion with PhysicsEngine (Layer 1)
  3. Predict intent with Layer 4c classifier
  4. Fill gaps with Layer 4b
  5. If video available: encode with CLIP → scene embedding → motion query

DROID dataset keys confirmed:
  steps/action                    → 7-dim float32 (6 cartesian + 1 gripper) at 15Hz
  steps/language_instruction      → text e.g. "Put the white book on top of the tissue box."
  steps/observation/wrist_image_left → JPEG bytes per frame
  episode_metadata/file_path      → episode ID

Usage:
    python3.9 experiments/layer5_scenario.py --parse
    python3.9 experiments/layer5_scenario.py --certify
    python3.9 experiments/layer5_scenario.py --chain
    python3.9 experiments/layer5_scenario.py --train
"""
import os, sys, json, struct, glob, argparse, re, time, random, io
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/S2S"))

import numpy as np

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

try:
    import clip as _clip
    CLIP_OK = True
except ImportError:
    CLIP_OK = False

try:
    from sentence_transformers import SentenceTransformer as _ST
    ST_OK = True
except ImportError:
    ST_OK = False

NPZ_PATH    = Path("experiments/sequences_real.npz")
DROID_DIR   = Path.home() / "droid_data"
MODEL_PATH  = Path("experiments/layer5_model.pt")
RESULTS_PATH = Path("experiments/results_layer5.json")

DROID_HZ    = 15       # DROID action frequency
WINDOW_SIZE = 30       # 2 seconds at 15Hz — same as RoboTurk in Layer 4


# ---------------------------------------------------------------------------
# DROID TFRecord parser — zero dependencies
# ---------------------------------------------------------------------------

def read_tfrecords(path, max_records=None):
    """Read raw TFRecord bytes. Returns list of raw protobuf bytes."""
    records = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)  # masked_crc32
            data = f.read(length)
            f.read(4)  # masked_crc32
            if len(data) < length:
                break
            records.append(data)
            if max_records and len(records) >= max_records:
                break
    return records


def parse_floats_after_key(raw, key, max_floats=10000):
    """
    Find key in protobuf bytes, read float32 list after it.
    TFRecord float_list: tag 0x52 or similar before the float data.
    """
    key_bytes = key.encode() if isinstance(key, str) else key
    pos = raw.find(key_bytes)
    if pos < 0:
        return []

    # After key, find float32 data — look for the float_list protobuf field
    chunk = raw[pos + len(key_bytes): pos + len(key_bytes) + max_floats * 4 + 100]

    # Float32 values are stored as little-endian IEEE 754
    # Find the float_list by looking for the length prefix
    # Scan for plausible float sequences (values in robot action range -5 to +5)
    floats = []
    i = 0
    while i + 4 <= len(chunk) and len(floats) < max_floats:
        val = struct.unpack_from("<f", chunk, i)[0]
        if -100.0 < val < 100.0 and val == val:  # finite and in range
            floats.append(val)
        i += 4
    return floats


def parse_text_after_key(raw, key):
    """Extract text string after a key in protobuf bytes."""
    key_bytes = key.encode() if isinstance(key, str) else key
    pos = raw.find(key_bytes)
    if pos < 0:
        return ""
    chunk = raw[pos + len(key_bytes): pos + len(key_bytes) + 500]
    # Find readable ASCII text
    matches = re.findall(b"[A-Za-z][A-Za-z0-9 _,.'!?-]{5,150}", chunk)
    for m in matches:
        text = m.decode("utf-8", errors="ignore").strip()
        # Skip key names themselves
        if text not in ("language_instruction", "steps", "action",
                        "observation", "wrist_image_left"):
            return text
    return ""


def parse_droid_episode(raw):
    """
    Parse one DROID TFRecord episode.
    Returns dict with action sequence, instruction, episode_id.
    """
    # Language instruction
    instruction = parse_text_after_key(raw, "steps/language_instruction")

    # Episode file path
    ep_id = parse_text_after_key(raw, "episode_metadata/file_path")

    # Action sequence — 7-dim per step (6 cartesian + 1 gripper)
    # Find steps/action then read float32 array
    action_pos = raw.find(b"steps/action")
    actions = []
    if action_pos >= 0:
        # Read float32 values after the key
        chunk_start = action_pos + len("steps/action")
        chunk = raw[chunk_start: chunk_start + 50000]

        # Parse protobuf float_list — floats stored in groups of 7
        raw_floats = []
        i = 0
        while i + 4 <= len(chunk) and len(raw_floats) < 5000:
            val = struct.unpack_from("<f", chunk, i)[0]
            if -50.0 < val < 50.0 and val == val:
                raw_floats.append(float(val))
            i += 4

        # Group into steps of 7 (6 cartesian + 1 gripper)
        for i in range(0, len(raw_floats) - 6, 7):
            step = raw_floats[i:i+7]
            if len(step) == 7:
                actions.append(step)

    return {
        "instruction": instruction,
        "episode_id":  ep_id,
        "actions":     actions,   # list of [x,y,z,rx,ry,rz,gripper]
        "n_steps":     len(actions),
    }


# ---------------------------------------------------------------------------
# S2S certification of DROID action sequences
# ---------------------------------------------------------------------------

def certify_episode(episode, engine):
    """
    Certify one DROID episode through PhysicsEngine.
    Actions are Cartesian deltas at 15Hz — treat as accel proxy.
    Returns certified windows with tiers.
    """
    actions = np.array(episode["actions"], dtype=np.float64)
    if len(actions) < WINDOW_SIZE:
        return []

    # Cartesian positions → pseudo-accel via double difference
    positions = np.cumsum(actions[:, :3], axis=0)
    if len(positions) > 2:
        vel   = np.diff(positions, axis=0) * DROID_HZ
        accel = np.diff(vel, axis=0) * DROID_HZ
    else:
        return []

    gyro_zero = np.zeros_like(accel)

    certified = []
    for i in range(0, len(accel) - WINDOW_SIZE, WINDOW_SIZE):
        chunk_a = accel[i:i+WINDOW_SIZE]
        chunk_g = gyro_zero[i:i+WINDOW_SIZE]

        ts = [int(j * 1e9 / DROID_HZ + random.gauss(0, 500000))
              for j in range(WINDOW_SIZE)]

        try:
            result = engine.certify(
                {"timestamps_ns": ts,
                 "accel": chunk_a.tolist(),
                 "gyro":  chunk_g.tolist()},
                segment="forearm"
            )
        except Exception:
            continue

        certified.append({
            "tier":        result.get("tier", "REJECTED"),
            "score":       result.get("physical_law_score", 0),
            "laws_passed": result.get("laws_passed", []),
            "window_idx":  i // WINDOW_SIZE,
            "instruction": episode["instruction"],
            "accel":       chunk_a.tolist(),
        })

    return certified


# ---------------------------------------------------------------------------
# Feature extraction (same 13-dim as Layer 4)
# ---------------------------------------------------------------------------

def extract_features(accel, hz=DROID_HZ):
    accel = np.array(accel, dtype=np.float64)
    if accel.ndim == 1:
        accel = accel.reshape(-1, 1)
    while accel.shape[1] < 3:
        accel = np.hstack([accel, np.zeros((len(accel), 1))])
    gyro = np.zeros_like(accel)

    f = []
    f.append(float(np.sqrt(np.mean(accel**2))))
    f.append(float(np.std(accel)))
    f.append(float(np.max(np.abs(accel))))
    f.append(float(np.sqrt(np.mean(gyro**2))))
    f.append(float(np.std(gyro)))

    if len(accel) > 3:
        jerk = np.diff(accel, n=1, axis=0) * hz
        f.append(float(np.sqrt(np.mean(jerk**2))))
        f.append(float(np.percentile(np.abs(jerk), 95)))
    else:
        f.extend([0.0, 0.0])

    for axis in range(3):
        fft   = np.abs(np.fft.rfft(accel[:, axis]))
        freqs = np.fft.rfftfreq(len(accel), 1/hz)
        f.append(float(freqs[np.argmax(fft)] if len(fft) > 0 else 0))

    c = np.corrcoef(accel[:, 0], accel[:, 1])[0, 1]
    f.append(float(c) if not np.isnan(c) else 0.0)
    f.append(float(np.linalg.norm(np.mean(accel, axis=0))))

    hist, _ = np.histogram(accel.flatten(), bins=20)
    hist = hist / (hist.sum() + 1e-10)
    f.append(float(-np.sum(hist * np.log(hist + 1e-10))))

    return np.array(f, dtype=np.float32)


# ---------------------------------------------------------------------------
# Full S2S chain on one episode
# ---------------------------------------------------------------------------

def run_full_chain(episode, engine, layer4c_model=None,
                   feat_mean=None, feat_std=None, labels=None):
    """
    Run Layers 1→4 on a DROID episode.
    Returns chain result dict.
    """
    result = {
        "instruction": episode["instruction"],
        "n_steps":     episode["n_steps"],
        "windows":     [],
        "chain_summary": {},
    }

    certified = certify_episode(episode, engine)
    if not certified:
        result["chain_summary"] = {"error": "no_certified_windows"}
        return result

    # Layer 1 summary
    tiers = [w["tier"] for w in certified]
    tier_counts = defaultdict(int)
    for t in tiers:
        tier_counts[t] += 1

    result["windows"] = certified
    result["chain_summary"]["layer1"] = {
        "n_windows":   len(certified),
        "tier_counts": dict(tier_counts),
        "cert_rate":   round((len(certified) - tier_counts["REJECTED"]) / max(len(certified), 1), 3),
    }

    # Layer 4a — predict next action for certified windows
    valid = [w for w in certified if w["tier"] != "REJECTED"]
    if valid:
        result["chain_summary"]["layer4a"] = {
            "n_valid_for_prediction": len(valid),
            "note": "next action prediction available via layer4_sequence_model.py",
        }

    # Layer 4c — classify intent from motion
    if layer4c_model is not None and feat_mean is not None and valid:
        import torch
        import torch.nn as nn

        feats = np.array([extract_features(w["accel"]) for w in valid[:10]],
                         dtype=np.float32)
        feats_norm = (feats - feat_mean) / feat_std
        with torch.no_grad():
            logits = layer4c_model(torch.tensor(feats_norm))
            top3   = torch.topk(logits.mean(0), 3)

        predicted = [(labels[i], float(p))
                     for i, p in zip(top3.indices.tolist(),
                                     torch.softmax(top3.values, 0).tolist())]
        result["chain_summary"]["layer4c"] = {
            "true_instruction": episode["instruction"],
            "predicted_intents": predicted,
        }

    return result


# ---------------------------------------------------------------------------
# Parse all DROID episodes
# ---------------------------------------------------------------------------

def parse_all(droid_dir=DROID_DIR, max_episodes=50):
    """Parse DROID TFRecords → list of episode dicts."""
    records = sorted(glob.glob(str(droid_dir / "**/*.tfrecord*"), recursive=True))
    if not records:
        records = sorted(glob.glob(str(droid_dir / "droid_100/**/*.tfrecord*"),
                                   recursive=True))
    print(f"Found {len(records)} TFRecord files")

    episodes = []
    for rec_path in records:
        try:
            raw_records = read_tfrecords(rec_path, max_records=None)
            for raw in raw_records:
                ep = parse_droid_episode(raw)
                BLOCKLIST = {
                    "joint_position", "joint_positio", "is_last", "is_terminal",
                    "is_first", "action_dict", "discount", "exterior_image_1_left",
                    "exterior_image_2_left", "wrist_image_left", "gripper_position",
                    "gripper_p", "language_instruction_2", "language_instruction_3",
                    "episode_metadata", "steps", "observation", "language_instruction",
                }
                if (ep["instruction"] and ep["n_steps"] > WINDOW_SIZE
                        and ep["instruction"] not in BLOCKLIST
                        and len(ep["instruction"]) > 10):
                    episodes.append(ep)
                    if len(episodes) >= max_episodes:
                        break
        except Exception as e:
            print(f"  {Path(rec_path).name}: error — {e}")
        if len(episodes) >= max_episodes:
            break

    print(f"Parsed {len(episodes)} valid episodes")
    if episodes:
        print(f"Sample instructions:")
        for ep in episodes[:5]:
            print(f"  [{ep['n_steps']} steps] {ep['instruction']}")
    return episodes


# ---------------------------------------------------------------------------
# Certify all episodes
# ---------------------------------------------------------------------------

def certify_all(episodes, out_dir="droid_certified"):
    engine   = PhysicsEngine()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    tier_counts   = defaultdict(int)
    instr_cert    = []

    print(f"\nCertifying {len(episodes)} DROID episodes...")
    print(f"{'Episode':>8}  {'Steps':>6}  {'Windows':>8}  "
          f"{'GOLD':>6}  {'BRNZ':>6}  {'REJ':>6}  {'Instruction'}")
    print("─" * 80)

    for i, ep in enumerate(episodes):
        certified = certify_episode(ep, engine)
        if not certified:
            continue

        tiers = [w["tier"] for w in certified]
        tc    = defaultdict(int)
        for t in tiers:
            tc[t] += 1
            tier_counts[t] += 1
        total_windows += len(certified)

        cert_rate = (len(certified) - tc["REJECTED"]) / max(len(certified), 1)
        instr_cert.append({
            "instruction": ep["instruction"],
            "cert_rate":   round(cert_rate, 3),
            "n_windows":   len(certified),
        })

        print(f"{i:>8}  {ep['n_steps']:>6}  {len(certified):>8}  "
              f"{tc.get('GOLD',0):>6}  {tc.get('BRONZE',0):>6}  "
              f"{tc.get('REJECTED',0):>6}  "
              f"{ep['instruction'][:50]}")

        # Save certified windows
        (out_path / f"ep_{i:04d}.json").write_text(
            json.dumps(certified[:20], indent=2))  # save first 20 windows

    n_cert = total_windows - tier_counts.get("REJECTED", 0)
    print("─" * 80)
    print(f"Total windows: {total_windows}  Certified: {n_cert}  "
          f"({100*n_cert//max(total_windows,1)}%)")
    print(f"GOLD: {tier_counts.get('GOLD',0)}  "
          f"SILVER: {tier_counts.get('SILVER',0)}  "
          f"BRONZE: {tier_counts.get('BRONZE',0)}  "
          f"REJECTED: {tier_counts.get('REJECTED',0)}")

    summary = {
        "dataset": "DROID-100",
        "n_episodes": len(episodes),
        "total_windows": total_windows,
        "certified": n_cert,
        "cert_rate": round(n_cert / max(total_windows, 1), 3),
        "tier_counts": dict(tier_counts),
        "instructions_sample": instr_cert[:10],
    }
    (out_path / "droid_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out_path}/droid_summary.json")
    return summary


# ---------------------------------------------------------------------------
# Full chain — all layers on all episodes
# ---------------------------------------------------------------------------

def run_chain_all(episodes, max_ep=20):
    engine = PhysicsEngine()

    # Load Layer 4c if available
    layer4c_model, feat_mean, feat_std, labels = None, None, None, None
    layer4c_path = Path("experiments/layer4c_model.pt")
    if layer4c_path.exists():
        try:
            import torch
            ckpt = torch.load(str(layer4c_path), map_location="cpu")

            import torch.nn as nn
            cfg = ckpt["config"]

            class MotionClassifier(nn.Module):
                def __init__(self, input_dim, hidden, n_labels, proj_dim=384):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
                        nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                        nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hidden, hidden // 2), nn.ReLU(),
                    )
                    self.classifier = nn.Linear(hidden // 2, n_labels)
                    self.projector  = nn.Sequential(
                        nn.Linear(hidden // 2, proj_dim), nn.LayerNorm(proj_dim))

                def forward(self, x):
                    return self.classifier(self.encoder(x.float()))

            layer4c_model = MotionClassifier(
                input_dim=cfg["input_dim"],
                hidden=cfg.get("hidden", 256),
                n_labels=cfg["n_labels"])
            layer4c_model.load_state_dict(ckpt["model_state"])
            layer4c_model.eval()
            feat_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
            feat_std  = np.array(ckpt["feat_std"],  dtype=np.float32)
            labels    = ckpt["unique_labels"]
            print("Layer 4c loaded")
        except Exception as e:
            print(f"Layer 4c not loaded: {e}")

    results = []
    print(f"\nRunning full S2S chain on {min(max_ep, len(episodes))} episodes...\n")

    for i, ep in enumerate(episodes[:max_ep]):
        r = run_full_chain(ep, engine, layer4c_model, feat_mean, feat_std, labels)
        results.append(r)

        s = r["chain_summary"]
        l1 = s.get("layer1", {})
        l4c = s.get("layer4c", {})

        print(f"Episode {i}: '{ep['instruction'][:50]}'")
        print(f"  Layer 1: {l1.get('n_windows',0)} windows, "
              f"{l1.get('cert_rate',0)*100:.0f}% certified")
        if l4c:
            print(f"  Layer 4c predicted: {l4c.get('predicted_intents', [])[:2]}")
        print()

    Path("experiments/results_layer5.json").write_text(
        json.dumps(results[:10], indent=2, default=str))
    print("Saved → experiments/results_layer5.json")
    return results


# ---------------------------------------------------------------------------
# CLIP scene understanding
# ---------------------------------------------------------------------------

def clip_scene(droid_dir, max_episodes=5):
    """
    Full Layer 5 chain: video frame + language instruction → CLIP scene match.
    For each episode: extract first JPEG frame, encode with CLIP,
    encode instruction with sentence-transformers, compute similarity.
    """
    if not CLIP_OK:
        print("CLIP not installed. Run: pip3.9 install git+https://github.com/openai/CLIP.git")
        return
    if not ST_OK:
        print("sentence-transformers not installed.")
        return

    import torch
    import torch.nn as nn

    print("Loading CLIP ViT-B/32...")
    clip_model, preprocess = _clip.load("ViT-B/32", device="cpu")
    clip_model.eval()

    print("Loading sentence-transformers...")
    st_model = _ST("all-MiniLM-L6-v2")

    records = sorted(glob.glob(str(droid_dir / "**/*.tfrecord*"), recursive=True))
    if not records:
        records = sorted(glob.glob(str(droid_dir / "droid_100/**/*.tfrecord*"),
                                   recursive=True))

    BLOCKLIST = {
        "joint_position", "joint_positio", "is_last", "is_terminal",
        "is_first", "action_dict", "discount", "exterior_image_1_left",
        "exterior_image_2_left", "wrist_image_left", "gripper_position",
        "gripper_p", "language_instruction_2", "language_instruction_3",
        "episode_metadata", "steps", "observation",
    }

    print(f"\nLayer 5 — CLIP Scene Understanding")
    print("=" * 65)
    print(f"{'Instruction':<45} {'CLIP sim':>10}")
    print("─" * 65)

    results = []
    ep_count = 0

    for rec_path in records:
        if ep_count >= max_episodes:
            break
        try:
            raw_records = read_tfrecords(rec_path, max_records=3)
            for raw in raw_records:
                if ep_count >= max_episodes:
                    break

                # Get instruction
                instruction = parse_text_after_key(raw, "steps/language_instruction")
                if not instruction or instruction in BLOCKLIST or len(instruction) < 10:
                    continue

                # Extract first JPEG frame
                pos, frame_bytes = 0, None
                while pos < len(raw) - 3:
                    if raw[pos:pos+3] == b"\xFF\xD8\xFF":
                        end = raw.find(b"\xFF\xD9", pos+2)
                        if end > pos and end - pos > 5000:
                            frame_bytes = raw[pos:end+2]
                            break
                        pos += 1
                    else:
                        pos += 1

                if frame_bytes is None:
                    continue

                try:
                    from PIL import Image as _Image
                    img = preprocess(
                        _Image.open(io.BytesIO(frame_bytes))
                    ).unsqueeze(0)
                except Exception:
                    continue

                # CLIP encode image
                with torch.no_grad():
                    img_emb   = clip_model.encode_image(img)
                    img_emb  /= img_emb.norm(dim=-1, keepdim=True)

                    # CLIP encode instruction text
                    text_tok  = _clip.tokenize([instruction[:77]])
                    text_emb  = clip_model.encode_text(text_tok)
                    text_emb /= text_emb.norm(dim=-1, keepdim=True)

                    scene_sim = float((img_emb @ text_emb.T)[0][0])

                # sentence-transformers encode for Layer 4c
                st_emb = st_model.encode([instruction])[0]

                print(f"{instruction[:45]:<45} {scene_sim:>10.4f}")

                results.append({
                    "instruction":  instruction,
                    "clip_sim":     round(scene_sim, 4),
                    "img_emb_dim":  img_emb.shape[-1],
                    "st_emb_dim":   len(st_emb),
                })
                ep_count += 1

        except Exception as e:
            print(f"  {Path(rec_path).name}: {e}")

    print("─" * 65)
    if results:
        mean_sim = sum(r["clip_sim"] for r in results) / len(results)
        print(f"Mean CLIP similarity: {mean_sim:.4f}  ({len(results)} episodes)")
        print(f"\nLayer 5 chain confirmed:")
        print(f"  Video frame  → CLIP ViT-B/32  → {results[0]['img_emb_dim']}-dim image embedding")
        print(f"  Instruction  → sentence-transformers → {results[0]['st_emb_dim']}-dim text embedding")
        print(f"  Scene match  → CLIP cosine similarity → {mean_sim:.4f} mean")
        print(f"  → Ready to route to Layer 4c intent retrieval")

    Path("experiments/results_layer5_clip.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nSaved → experiments/results_layer5_clip.json")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S2S Layer 5 — DROID Scenario")
    parser.add_argument("--parse",   action="store_true",
                        help="Parse DROID TFRecords and show instructions")
    parser.add_argument("--certify", action="store_true",
                        help="Certify all episodes through PhysicsEngine")
    parser.add_argument("--chain",   action="store_true",
                        help="Run full Layer 1→4 chain on episodes")
    parser.add_argument("--clip",    action="store_true",
                        help="Run CLIP scene understanding on video frames")
    parser.add_argument("--max",     type=int, default=50)
    parser.add_argument("--droid",   default=str(DROID_DIR))
    args = parser.parse_args()

    droid_dir = Path(args.droid)
    if not droid_dir.exists():
        # Try subdirectory
        sub = droid_dir / "droid_100" / "1.0.0"
        if sub.exists():
            droid_dir = sub
        else:
            print(f"DROID data not found at {droid_dir}")
            sys.exit(1)

    if args.parse:
        episodes = parse_all(droid_dir, max_episodes=args.max)
        print(f"\nTotal valid episodes: {len(episodes)}")
        unique_instructions = list(set(ep["instruction"] for ep in episodes))
        print(f"Unique instructions: {len(unique_instructions)}")
        for instr in unique_instructions[:10]:
            print(f"  {instr}")

    elif args.certify:
        episodes = parse_all(droid_dir, max_episodes=args.max)
        certify_all(episodes)

    elif args.chain:
        episodes = parse_all(droid_dir, max_episodes=args.max)
        run_chain_all(episodes, max_ep=args.max)

    elif args.clip:
        clip_scene(droid_dir, max_episodes=args.max)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
