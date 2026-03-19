#!/usr/bin/env python3
"""
s2s_api_v1_3.py — S2S REST API Server v1.3 (stdlib only, zero dependencies)

All endpoints:
  GET  /health                    — health check + active sessions
  GET  /version                   — version + available sensors
  GET  /sessions                  — list active streaming sessions
  GET  /stream/{session_id}/status — session stats + last cert

  POST /certify/imu               — certify a batch of IMU frames
  POST /certify/emg               — certify a batch of EMG frames
  POST /certify/lidar             — certify a batch of LiDAR frames (scalar or pointcloud)
  POST /certify/thermal           — certify a batch of thermal frames
  POST /certify/ppg               — certify a batch of PPG frames
  POST /certify/fusion            — fuse multiple stream certs → unified cert
  POST /stream/frame              — push a single frame to a named persistent session

  DELETE /stream/{session_id}     — end a streaming session

Signing middleware (optional):
  Set --sign-key path/to/device.private.pem to auto-sign all outgoing certs.

Registry middleware (optional):
  Set --registry path/to/registry.json to validate incoming device_id fields.

Usage:
  python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080
  python3 -m s2s_standard_v1_3.s2s_api_v1_3 --port 8080 \\
      --sign-key keys/server.private.pem \\
      --registry registry.json
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from .s2s_stream_certify_v1_3  import StreamCertifier
    from .s2s_emg_certify_v1_3     import EMGStreamCertifier, certify_emg_channels
    from .s2s_lidar_certify_v1_3   import (LiDARStreamCertifier,
                                            analyze_scalar_lidar,
                                            analyze_pointcloud_lidar)
    from .s2s_thermal_certify_v1_3 import ThermalStreamCertifier, certify_thermal_frames
    from .s2s_ppg_certify_v1_3     import PPGStreamCertifier, certify_ppg_channels
    from .s2s_fusion_v1_3          import FusionCertifier
    from .s2s_signing_v1_3         import CertSigner, attach_signer_to_certifier
    from .s2s_registry_v1_3        import RegistryValidator
    from .constants                 import (STREAM_WINDOW_DEFAULT, STREAM_STEP_DEFAULT,
                                            VERSION_MAJOR, VERSION_MINOR)
except Exception:
    from s2s_stream_certify_v1_3  import StreamCertifier
    from s2s_emg_certify_v1_3     import EMGStreamCertifier, certify_emg_channels
    from s2s_lidar_certify_v1_3   import (LiDARStreamCertifier,
                                           analyze_scalar_lidar,
                                           analyze_pointcloud_lidar)
    from s2s_thermal_certify_v1_3 import ThermalStreamCertifier, certify_thermal_frames
    from s2s_ppg_certify_v1_3     import PPGStreamCertifier, certify_ppg_channels
    from s2s_fusion_v1_3          import FusionCertifier
    from s2s_signing_v1_3         import CertSigner, attach_signer_to_certifier
    from s2s_registry_v1_3        import RegistryValidator
    from constants                 import (STREAM_WINDOW_DEFAULT, STREAM_STEP_DEFAULT,
                                           VERSION_MAJOR, VERSION_MINOR)

API_VERSION = "1.3.0"

# ---------------------------------------------------------------------------
# Global middleware (set at startup)
# ---------------------------------------------------------------------------
_signer:    Optional[CertSigner]        = None
_validator: Optional[RegistryValidator] = None


def _sign(cert: Dict[str, Any]) -> Dict[str, Any]:
    return _signer.sign_cert(cert) if _signer else cert


def _registry_check(device_id: Optional[str]) -> Tuple[bool, str]:
    if not _validator or not device_id:
        return True, "OK"
    d = _validator.registry.get(device_id)
    if not d:
        return False, f"DEVICE_NOT_REGISTERED: {device_id}"
    if d.get("revoked"):
        return False, "DEVICE_REVOKED"
    return True, "OK"


# ---------------------------------------------------------------------------
# Session registry (thread-safe, in-memory)
# ---------------------------------------------------------------------------

class SessionRegistry:
    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self._sessions: Dict[str, Any] = {}
        self._meta:     Dict[str, Dict[str, Any]] = {}

    def get_or_create(self, session_id: str, sensor_type: str,
                      sensor_names: Optional[List[str]] = None,
                      sampling_hz: float = 240.0,
                      window: int = STREAM_WINDOW_DEFAULT,
                      step: int = STREAM_STEP_DEFAULT,
                      device_id: str = "unknown",
                      lidar_mode: str = "scalar",
                      thermal_width: int = 32,
                      thermal_height: int = 24) -> Any:
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            stype = sensor_type.lower()
            if stype == "emg":
                c = EMGStreamCertifier(
                    n_channels=len(sensor_names) if sensor_names else 8,
                    sampling_hz=max(sampling_hz, 500.0),
                    channel_names=sensor_names, window=window, step=step,
                    device_id=device_id, session_id=session_id)
            elif stype == "lidar":
                c = LiDARStreamCertifier(mode=lidar_mode, window=window,
                                         step=step, device_id=device_id)
            elif stype == "thermal":
                c = ThermalStreamCertifier(
                    frame_width=thermal_width, frame_height=thermal_height,
                    window=window, step=step,
                    device_id=device_id, session_id=session_id)
            elif stype == "ppg":
                c = PPGStreamCertifier(
                    n_channels=len(sensor_names) if sensor_names else 2,
                    sampling_hz=max(sampling_hz, 25.0),
                    channel_names=sensor_names, window=window, step=step,
                    device_id=device_id, session_id=session_id)
            else:
                names = sensor_names or ["accel_x","accel_y","accel_z",
                                         "gyro_x","gyro_y","gyro_z"]
                c = StreamCertifier(sensor_names=names, window=window,
                                    step=step, device_id=device_id)
            if _signer and hasattr(c, "_evaluate_window"):
                attach_signer_to_certifier(c, _signer)
            self._sessions[session_id] = c
            self._meta[session_id] = {
                "created_at": time.time(), "sensor_type": stype,
                "device_id": device_id, "certs_buffer": []}
            return c

    def push_frame(self, session_id: str, ts_ns: int,
                   values: List[float], **kw) -> Optional[Dict[str, Any]]:
        c = self.get_or_create(session_id, **kw)
        with self._lock:
            result = c.push_frame(ts_ns, values)
            if result:
                buf = self._meta[session_id]["certs_buffer"]
                buf.append(result)
                if len(buf) > 50: buf.pop(0)
            return result

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if session_id not in self._sessions: return None
            meta = self._meta[session_id]
            return {
                "session_id": session_id,
                "stats":      self._sessions[session_id].stats,
                "meta":       {k: v for k, v in meta.items() if k != "certs_buffer"},
                "last_cert":  meta["certs_buffer"][-1] if meta["certs_buffer"] else None,
            }

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                del self._meta[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        with self._lock: return list(self._sessions.keys())


_registry = SessionRegistry()


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------

def _resp(handler, code: int, body: Any) -> None:
    payload = json.dumps(body, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)


def _read_body(handler) -> Optional[Dict[str, Any]]:
    n = int(handler.headers.get("Content-Length", 0))
    if not n: return {}
    try: return json.loads(handler.rfile.read(n).decode("utf-8"))
    except: return None


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class S2SHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args): pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path.rstrip("/")

        if p == "/health":
            _resp(self, 200, {
                "status": "ok", "api_version": API_VERSION,
                "active_sessions": len(_registry.list_sessions()),
                "signing_enabled": _signer is not None,
                "registry_enabled": _validator is not None,
                "timestamp_ns": time.time_ns(),
            })
        elif p == "/version":
            _resp(self, 200, {
                "api_version": API_VERSION,
                "s2s_version": f"{VERSION_MAJOR}.{VERSION_MINOR}",
                "signing_mode": _signer.mode if _signer else "disabled",
                "signing_key_id": _signer.key_id if _signer else None,
                "registry": _validator.registry.summary() if _validator else None,
                "sensors": ["imu","emg","lidar","thermal","ppg","fusion"],
                "endpoints": [
                    "GET  /health", "GET  /version", "GET  /sessions",
                    "GET  /stream/{id}/status",
                    "POST /certify/imu", "POST /certify/emg",
                    "POST /certify/lidar", "POST /certify/thermal",
                    "POST /certify/ppg", "POST /certify/fusion",
                    "POST /stream/frame", "DELETE /stream/{id}",
                ],
            })
        elif p == "/sessions":
            _resp(self, 200, {"sessions": _registry.list_sessions()})
        elif p.startswith("/stream/") and p.count("/") == 2:
            sid = p.split("/")[2]
            s = _registry.get_status(sid)
            _resp(self, 200 if s else 404, s or {"error": f"Session '{sid}' not found"})
        else:
            _resp(self, 404, {"error": "Not found"})

    def do_DELETE(self):
        p = urlparse(self.path).path.rstrip("/")
        if p.startswith("/stream/") and p.count("/") == 2:
            sid = p.split("/")[2]
            ok  = _registry.delete(sid)
            _resp(self, 200 if ok else 404, {"deleted": ok, "session_id": sid})
        else:
            _resp(self, 404, {"error": "Not found"})

    def do_POST(self):
        p    = urlparse(self.path).path.rstrip("/")
        body = _read_body(self)
        if body is None:
            _resp(self, 400, {"error": "Invalid JSON"}); return
        try:
            self._post(p, body)
        except Exception as e:
            _resp(self, 500, {"error": str(e), "type": type(e).__name__})

    def _post(self, p: str, b: Dict[str, Any]) -> None:

        # ---- /certify/imu -----------------------------------------------
        if p == "/certify/imu":
            frames = b.get("frames", [])
            dev    = b.get("device_id", "unknown")
            if not frames: _resp(self, 400, {"error": "No frames"}); return
            ok, why = _registry_check(dev)
            if not ok: _resp(self, 403, {"error": why}); return

            names = b.get("sensor_names") or [
                "accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
            sc   = StreamCertifier(sensor_names=names,
                                   window=min(b.get("window", STREAM_WINDOW_DEFAULT), len(frames)),
                                   step=len(frames), device_id=dev)
            cert = None
            for f in frames:
                r = sc.push_frame(int(f["ts_ns"]), [float(v) for v in f["values"]])
                if r: cert = r
            _resp(self, 200, _sign(cert) if cert else {
                "status": "PENDING", "buffer_fill": sc.buffer_fill, "received": len(frames)})

        # ---- /certify/emg -----------------------------------------------
        elif p == "/certify/emg":
            frames = b.get("frames", [])
            dev    = b.get("device_id", "unknown")
            if not frames: _resp(self, 400, {"error": "No frames"}); return
            ok, why = _registry_check(dev)
            if not ok: _resp(self, 403, {"error": why}); return

            hz     = float(b.get("sampling_hz", 1000.0))
            n_ch   = len(frames[0].get("values", []))
            names  = b.get("channel_names") or [f"emg_{i}" for i in range(n_ch)]
            ts     = [int(f["ts_ns"]) for f in frames]
            chs    = [[] for _ in range(n_ch)]
            for f in frames:
                vs = f.get("values", [])
                for i in range(n_ch):
                    chs[i].append(float(vs[i]) if i < len(vs) else math.nan)
            _resp(self, 200, _sign(certify_emg_channels(
                names=names, channels=chs, timestamps_ns=ts,
                sampling_hz=hz, device_id=dev)))

        # ---- /certify/lidar ---------------------------------------------
        elif p == "/certify/lidar":
            frames = b.get("frames", [])
            dev    = b.get("device_id", "unknown")
            mode   = b.get("mode", "scalar")
            if not frames: _resp(self, 400, {"error": "No frames"}); return
            ok, why = _registry_check(dev)
            if not ok: _resp(self, 403, {"error": why}); return

            ts = [int(f["ts_ns"]) for f in frames]
            if mode == "pointcloud":
                cert = analyze_pointcloud_lidar(
                    frames_xyz=[f.get("values", []) for f in frames],
                    timestamps_ns=ts, device_id=dev)
            else:
                cert = analyze_scalar_lidar(
                    distances=[f.get("values", [0.0])[0] for f in frames],
                    timestamps_ns=ts, device_id=dev)
            cert.update({
                "frame_start_ts_ns": ts[0], "frame_end_ts_ns": ts[-1],
                "duration_ms": (ts[-1] - ts[0]) / 1e6})
            _resp(self, 200, _sign(cert))

        # ---- /certify/thermal -------------------------------------------
        elif p == "/certify/thermal":
            frames = b.get("frames", [])
            dev    = b.get("device_id", "unknown")
            w, h   = int(b.get("width", 32)), int(b.get("height", 24))
            if not frames: _resp(self, 400, {"error": "No frames"}); return
            ok, why = _registry_check(dev)
            if not ok: _resp(self, 403, {"error": why}); return

            ts     = [int(f["ts_ns"]) for f in frames]
            pixels = [f.get("pixels", f.get("values", [])) for f in frames]
            _resp(self, 200, _sign(certify_thermal_frames(
                frames_flat=pixels, timestamps_ns=ts,
                width=w, height=h, device_id=dev)))

        # ---- /certify/ppg -----------------------------------------------
        elif p == "/certify/ppg":
            frames = b.get("frames", [])
            dev    = b.get("device_id", "unknown")
            if not frames: _resp(self, 400, {"error": "No frames"}); return
            ok, why = _registry_check(dev)
            if not ok: _resp(self, 403, {"error": why}); return

            hz    = float(b.get("sampling_hz", 100.0))
            n_ch  = len(frames[0].get("values", []))
            names = b.get("channel_names") or [f"ppg_{i}" for i in range(n_ch)]
            ts    = [int(f["ts_ns"]) for f in frames]
            chs   = [[] for _ in range(n_ch)]
            for f in frames:
                vs = f.get("values", [])
                for i in range(n_ch):
                    chs[i].append(float(vs[i]) if i < len(vs) else math.nan)
            _resp(self, 200, _sign(certify_ppg_channels(
                names=names, channels=chs, timestamps_ns=ts,
                sampling_hz=hz, device_id=dev)))

        # ---- /certify/fusion --------------------------------------------
        elif p == "/certify/fusion":
            dev  = b.get("device_id", "unknown")
            sid  = b.get("session_id")
            cmap = {k: b.get(f"{k}_cert")
                    for k in ["imu","emg","lidar","thermal","ppg"]}
            provided = sum(1 for c in cmap.values() if c) + len(b.get("extra_certs", []))
            if provided < 2:
                _resp(self, 400, {"error": "Need at least 2 stream certs"}); return

            fc = FusionCertifier(device_id=dev, session_id=sid)
            if cmap["imu"]:     fc.add_imu_cert(cmap["imu"])
            if cmap["emg"]:     fc.add_emg_cert(cmap["emg"])
            if cmap["lidar"]:   fc.add_stream("lidar",   cmap["lidar"],   "LIDAR")
            if cmap["thermal"]: fc.add_stream("thermal", cmap["thermal"], "THERMAL")
            if cmap["ppg"]:     fc.add_stream("ppg",     cmap["ppg"],     "PPG")
            for ec in b.get("extra_certs", []):
                fc.add_stream(ec["stream_id"], ec["cert"], ec.get("sensor_type"))
            _resp(self, 200, _sign(fc.certify()))

        # ---- /stream/frame ----------------------------------------------
        elif p == "/stream/frame":
            sid    = b.get("session_id", "default")
            dev    = b.get("device_id",  "unknown")
            ts_ns  = int(b.get("ts_ns", time.time_ns()))
            values = [float(v) for v in b.get("values", [])]
            if not values: _resp(self, 400, {"error": "No values"}); return
            ok, why = _registry_check(dev)
            if not ok: _resp(self, 403, {"error": why}); return

            result = _registry.push_frame(
                session_id=sid, ts_ns=ts_ns, values=values,
                sensor_type=b.get("sensor_type", "imu"),
                sensor_names=b.get("sensor_names"),
                sampling_hz=float(b.get("sampling_hz", 240.0)),
                window=int(b.get("window", STREAM_WINDOW_DEFAULT)),
                step=int(b.get("step",   STREAM_STEP_DEFAULT)),
                device_id=dev,
                lidar_mode=b.get("lidar_mode", "scalar"),
                thermal_width=int(b.get("thermal_width",  32)),
                thermal_height=int(b.get("thermal_height", 24)),
            )
            if result:
                _resp(self, 200, result)
            else:
                s = _registry.get_status(sid)
                _resp(self, 202, {
                    "status": "BUFFERING", "session_id": sid,
                    "buffer_fill": s["stats"]["buffer_fill"] if s else 0,
                    "window":      s["stats"]["window"]      if s else b.get("window"),
                })
        else:
            _resp(self, 404, {"error": "Not found", "path": p})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2S v1.3 REST API Server",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--port",     type=int, default=8080)
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--sign-key", default=None,
                   help="Server private .pem — auto-signs all outgoing certs")
    p.add_argument("--registry", default=None,
                   help="Device registry JSON — validates incoming device_ids")
    p.add_argument("--quiet",    action="store_true")
    return p.parse_args()


def main() -> None:
    global _signer, _validator
    args = parse_args()

    if args.sign_key:
        try:
            _signer = CertSigner.from_pem_file(args.sign_key)
            if not args.quiet:
                print(json.dumps({"event":"signing_enabled",
                                  "mode": _signer.mode, "key_id": _signer.key_id}))
        except Exception as e:
            print(json.dumps({"event":"signing_error","error":str(e)}))

    if args.registry:
        try:
            _validator = RegistryValidator(args.registry)
            if not args.quiet:
                print(json.dumps({"event":"registry_enabled",
                                  "summary": _validator.registry.summary()}))
        except Exception as e:
            print(json.dumps({"event":"registry_error","error":str(e)}))

    server = HTTPServer((args.host, args.port), S2SHandler)

    if not args.quiet:
        print(json.dumps({
            "event": "api_start", "host": args.host, "port": args.port,
            "api_version": API_VERSION,
            "signing":  _signer.key_id if _signer else "disabled",
            "registry": args.registry or "disabled",
            "sensors":  ["imu","emg","lidar","thermal","ppg","fusion"],
        }, indent=2))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if not args.quiet: print("\n[S2S API] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
