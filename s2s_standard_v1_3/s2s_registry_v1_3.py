#!/usr/bin/env python3
"""
s2s_registry_v1_3.py — S2S Device Registry & Whitelist Manager

Maps device IDs to:
  - Expected hardware jitter signature (vendor RMS jitter specs)
  - Public key (for cert signature verification)
  - Sensor profile (which sensors the device has, their expected Hz)
  - Trust tier (TRUSTED / PROVISIONAL / UNTRUSTED)
  - Ancestry / lineage info (who minted the device, when, for what purpose)

This is the foundation for the marketplace:
  - Buyers verify a dataset came from a registered, trusted device
  - Royalties flow back to the original device owner / data collector
  - Forged certs are caught because the device's public key doesn't match

Registry is stored as a JSON file (local) or can be extended to a REST backend.

Usage:
  # Create registry:
  from s2s_standard_v1_3.s2s_registry_v1_3 import DeviceRegistry
  reg = DeviceRegistry("registry.json")

  # Register a device:
  reg.register(
      device_id="glove_v2_001",
      sensor_profile="imu_9dof",
      expected_jitter_ns=4500.0,
      public_key_pem="...",   # from s2s_signing_v1_3 keygen
      owner="timur@scan2sell.io",
      trust_tier="TRUSTED",
  )

  # Validate a certificate:
  ok, reason, device = reg.validate_cert(cert_dict)

  # CLI:
  python3 -m s2s_standard_v1_3.s2s_registry_v1_3 register --device-id glove_001 --profile imu_9dof
  python3 -m s2s_standard_v1_3.s2s_registry_v1_3 list
  python3 -m s2s_standard_v1_3.s2s_registry_v1_3 validate cert.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .constants import SENSOR_PROFILES, VERSION_MAJOR, VERSION_MINOR
    from .s2s_signing_v1_3 import CertVerifier, ED25519_AVAILABLE
except Exception:
    from constants import SENSOR_PROFILES, VERSION_MAJOR, VERSION_MINOR
    from s2s_signing_v1_3 import CertVerifier, ED25519_AVAILABLE

# ---------------------------------------------------------------------------
# Registry constants
# ---------------------------------------------------------------------------
TRUST_TRUSTED      = "TRUSTED"       # verified device, public key on file
TRUST_PROVISIONAL  = "PROVISIONAL"   # device registered but not yet verified
TRUST_UNTRUSTED    = "UNTRUSTED"     # explicitly flagged / revoked
TRUST_TIERS        = {TRUST_TRUSTED, TRUST_PROVISIONAL, TRUST_UNTRUSTED}

JITTER_TOLERANCE_PCT_DEFAULT = 20.0  # ±20% tolerance on expected jitter

# ---------------------------------------------------------------------------
# Device record schema
# ---------------------------------------------------------------------------
# {
#   "device_id":           str,
#   "sensor_profile":      str,        # key into SENSOR_PROFILES
#   "expected_jitter_ns":  float,      # vendor-spec RMS jitter
#   "jitter_tolerance_pct": float,
#   "public_key_pem":      str | None, # Ed25519 public key PEM
#   "key_id":              str | None, # short fingerprint
#   "owner":               str,        # email / name / org
#   "trust_tier":          str,        # TRUSTED / PROVISIONAL / UNTRUSTED
#   "registered_at":       int,        # unix timestamp
#   "last_seen_at":        int | None,
#   "cert_count":          int,        # how many certs have been issued
#   "notes":               str,
#   "revoked":             bool,
#   "revoked_reason":      str | None,
# }


def _key_id_from_pem(pem: str) -> Optional[str]:
    """Derive short hex fingerprint from PEM public key."""
    try:
        raw = pem.encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DeviceRegistry
# ---------------------------------------------------------------------------

class DeviceRegistry:
    """
    JSON-backed device registry.

    Thread safety: Uses file-level locking via atomic replace writes.
    For concurrent multi-process usage, use a proper database instead.
    """

    def __init__(self, registry_path: str = "s2s_device_registry.json"):
        self._path   = Path(registry_path)
        self._data: Dict[str, Any] = {"devices": {}, "schema_version": "1.3"}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {"devices": {}, "schema_version": "1.3"}

    def _save(self) -> None:
        """Atomic save — write to temp then replace."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        device_id: str,
        sensor_profile: str,
        owner: str,
        expected_jitter_ns: float = 5000.0,
        jitter_tolerance_pct: float = JITTER_TOLERANCE_PCT_DEFAULT,
        public_key_pem: Optional[str] = None,
        trust_tier: str = TRUST_PROVISIONAL,
        notes: str = "",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Register a new device or update an existing one.
        Returns the device record.
        """
        if trust_tier not in TRUST_TIERS:
            raise ValueError(f"trust_tier must be one of {TRUST_TIERS}")

        if sensor_profile not in SENSOR_PROFILES:
            raise ValueError(
                f"Unknown sensor_profile '{sensor_profile}'. "
                f"Valid: {list(SENSOR_PROFILES.keys())}"
            )

        devices = self._data.setdefault("devices", {})

        if device_id in devices and not overwrite:
            raise ValueError(
                f"Device '{device_id}' already registered. "
                f"Use overwrite=True to update."
            )

        key_id = _key_id_from_pem(public_key_pem) if public_key_pem else None

        record: Dict[str, Any] = {
            "device_id":            device_id,
            "sensor_profile":       sensor_profile,
            "expected_jitter_ns":   expected_jitter_ns,
            "jitter_tolerance_pct": jitter_tolerance_pct,
            "public_key_pem":       public_key_pem,
            "key_id":               key_id,
            "owner":                owner,
            "trust_tier":           trust_tier,
            "registered_at":        int(time.time()),
            "last_seen_at":         None,
            "cert_count":           0,
            "notes":                notes,
            "revoked":              False,
            "revoked_reason":       None,
        }

        devices[device_id] = record
        self._save()
        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, device_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get("devices", {}).get(device_id)

    def all_devices(self) -> List[Dict[str, Any]]:
        return list(self._data.get("devices", {}).values())

    def trusted_devices(self) -> List[Dict[str, Any]]:
        return [d for d in self.all_devices() if d["trust_tier"] == TRUST_TRUSTED and not d["revoked"]]

    # ------------------------------------------------------------------
    # Certificate validation
    # ------------------------------------------------------------------

    def validate_cert(
        self,
        cert: Dict[str, Any],
        verify_signature: bool = True,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate a certificate against the registry.

        Checks:
          1. device_id is registered
          2. Device is not revoked
          3. Device trust tier is TRUSTED (warns for PROVISIONAL)
          4. Hardware jitter matches expected spec (if cert has jitter metrics)
          5. Cryptographic signature is valid (if device has a public key)

        Returns: (is_valid: bool, reason: str, device_record: dict | None)
        """
        device_id = cert.get("device_id") or cert.get("meta_in", {}).get("device_id")

        if not device_id:
            return False, "CERT_HAS_NO_DEVICE_ID", None

        device = self.get(device_id)
        if not device:
            return False, f"DEVICE_NOT_REGISTERED: {device_id}", None

        if device["revoked"]:
            reason = device.get("revoked_reason", "no reason given")
            return False, f"DEVICE_REVOKED: {reason}", device

        if device["trust_tier"] == TRUST_UNTRUSTED:
            return False, "DEVICE_MARKED_UNTRUSTED", device

        # Jitter fingerprint check (if cert has jitter metrics)
        metrics = cert.get("metrics", {})
        rms_jitter = metrics.get("rms_jitter_ns")
        if rms_jitter is not None and device["expected_jitter_ns"] > 0:
            expected  = device["expected_jitter_ns"]
            tolerance = device["jitter_tolerance_pct"] / 100.0
            margin    = max(expected * tolerance, 1.0)
            if abs(rms_jitter - expected) > margin:
                return (
                    False,
                    f"JITTER_MISMATCH: measured={rms_jitter:.1f}ns "
                    f"expected={expected:.1f}ns ±{margin:.1f}ns",
                    device,
                )

        # Signature verification
        if verify_signature and device.get("public_key_pem"):
            sig = cert.get("_signature")
            if not sig:
                return False, "CERT_UNSIGNED_BUT_DEVICE_HAS_KEY", device

            verifier = CertVerifier(
                public_key_bytes=device["public_key_pem"].encode("utf-8")
            )
            ok, sig_reason = verifier.verify_cert(cert)
            if not ok:
                return False, f"SIGNATURE_INVALID: {sig_reason}", device

        # Update last seen
        device["last_seen_at"] = int(time.time())
        device["cert_count"]   = device.get("cert_count", 0) + 1
        self._save()

        tier = device["trust_tier"]
        if tier == TRUST_PROVISIONAL:
            return True, "VALID_PROVISIONAL_DEVICE", device

        return True, "VALID_TRUSTED_DEVICE", device

    # ------------------------------------------------------------------
    # Trust management
    # ------------------------------------------------------------------

    def promote(self, device_id: str) -> bool:
        """Promote a PROVISIONAL device to TRUSTED."""
        device = self.get(device_id)
        if not device:
            return False
        device["trust_tier"] = TRUST_TRUSTED
        self._save()
        return True

    def revoke(self, device_id: str, reason: str = "manually revoked") -> bool:
        """Revoke a device — its future certs will be rejected."""
        device = self.get(device_id)
        if not device:
            return False
        device["revoked"]        = True
        device["revoked_reason"] = reason
        device["trust_tier"]     = TRUST_UNTRUSTED
        self._save()
        return True

    def update_public_key(self, device_id: str, public_key_pem: str) -> bool:
        """Update the public key for a device (key rotation)."""
        device = self.get(device_id)
        if not device:
            return False
        device["public_key_pem"] = public_key_pem
        device["key_id"]         = _key_id_from_pem(public_key_pem)
        self._save()
        return True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        devices = self.all_devices()
        return {
            "total":        len(devices),
            "trusted":      sum(1 for d in devices if d["trust_tier"] == TRUST_TRUSTED),
            "provisional":  sum(1 for d in devices if d["trust_tier"] == TRUST_PROVISIONAL),
            "revoked":      sum(1 for d in devices if d["revoked"]),
            "with_pubkey":  sum(1 for d in devices if d.get("public_key_pem")),
            "registry_path": str(self._path),
            "schema_version": self._data.get("schema_version", "unknown"),
        }

    def export_public_keys(self) -> Dict[str, str]:
        """
        Returns {device_id: public_key_pem} for all TRUSTED devices with keys.
        Useful for distributing to marketplace verifiers.
        """
        return {
            d["device_id"]: d["public_key_pem"]
            for d in self.trusted_devices()
            if d.get("public_key_pem")
        }


# ---------------------------------------------------------------------------
# Registry-aware cert validator (convenience wrapper)
# ---------------------------------------------------------------------------

class RegistryValidator:
    """
    Validates incoming stream certs against the registry.
    Intended to be used inside s2s_api_v1_3.py as middleware.

    Usage:
        rv = RegistryValidator("registry.json")
        ok, reason, device = rv.validate(cert_dict)
    """

    def __init__(self, registry_path: str = "s2s_device_registry.json"):
        self._registry = DeviceRegistry(registry_path)

    def validate(self, cert: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        return self._registry.validate_cert(cert)

    def is_trusted(self, device_id: str) -> bool:
        d = self._registry.get(device_id)
        return bool(d and d["trust_tier"] == TRUST_TRUSTED and not d.get("revoked"))

    def profile_for(self, device_id: str) -> Optional[Dict[str, Any]]:
        d = self._registry.get(device_id)
        if not d:
            return None
        return SENSOR_PROFILES.get(d["sensor_profile"])

    @property
    def registry(self) -> DeviceRegistry:
        return self._registry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S2S v1.3 — Device Registry Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--registry", default="s2s_device_registry.json",
                   help="Path to registry JSON file")
    sub = p.add_subparsers(dest="command", required=True)

    # register
    reg = sub.add_parser("register", help="Register a new device")
    reg.add_argument("--device-id",   required=True)
    reg.add_argument("--profile",     required=True, choices=list(SENSOR_PROFILES.keys()))
    reg.add_argument("--owner",       required=True, help="Owner email or name")
    reg.add_argument("--jitter-ns",   type=float, default=5000.0, dest="jitter_ns")
    reg.add_argument("--pubkey",      default=None, help="Path to .public.pem file")
    reg.add_argument("--trust",       default=TRUST_PROVISIONAL, choices=list(TRUST_TIERS))
    reg.add_argument("--notes",       default="")
    reg.add_argument("--overwrite",   action="store_true")

    # list
    sub.add_parser("list", help="List all registered devices")

    # validate
    val = sub.add_parser("validate", help="Validate a certificate JSON against registry")
    val.add_argument("cert_file", help="Path to certificate JSON")

    # promote
    pro = sub.add_parser("promote", help="Promote device to TRUSTED")
    pro.add_argument("device_id")

    # revoke
    rev = sub.add_parser("revoke", help="Revoke a device")
    rev.add_argument("device_id")
    rev.add_argument("--reason", default="manually revoked")

    # summary
    sub.add_parser("summary", help="Show registry summary stats")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    reg  = DeviceRegistry(args.registry)

    if args.command == "register":
        pubkey_pem = None
        if args.pubkey:
            pubkey_pem = Path(args.pubkey).read_text()
        record = reg.register(
            device_id           = args.device_id,
            sensor_profile      = args.profile,
            owner               = args.owner,
            expected_jitter_ns  = args.jitter_ns,
            public_key_pem      = pubkey_pem,
            trust_tier          = args.trust,
            notes               = args.notes,
            overwrite           = args.overwrite,
        )
        print(json.dumps({"event": "registered", "record": record}, indent=2, default=str))

    elif args.command == "list":
        devices = reg.all_devices()
        for d in devices:
            print(json.dumps({
                "device_id":   d["device_id"],
                "profile":     d["sensor_profile"],
                "trust_tier":  d["trust_tier"],
                "owner":       d["owner"],
                "cert_count":  d["cert_count"],
                "revoked":     d["revoked"],
                "key_id":      d.get("key_id"),
            }))
        if not devices:
            print("(no devices registered)")

    elif args.command == "validate":
        cert = json.loads(Path(args.cert_file).read_text())
        ok, reason, device = reg.validate_cert(cert)
        print(json.dumps({
            "valid":     ok,
            "reason":    reason,
            "device_id": device["device_id"] if device else None,
            "trust_tier": device["trust_tier"] if device else None,
        }, indent=2))

    elif args.command == "promote":
        ok = reg.promote(args.device_id)
        print(json.dumps({"promoted": ok, "device_id": args.device_id}))

    elif args.command == "revoke":
        ok = reg.revoke(args.device_id, args.reason)
        print(json.dumps({"revoked": ok, "device_id": args.device_id, "reason": args.reason}))

    elif args.command == "summary":
        print(json.dumps(reg.summary(), indent=2))


if __name__ == "__main__":
    main()
