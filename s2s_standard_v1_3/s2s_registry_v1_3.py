#!/usr/bin/env python3
"""
s2s_registry_v1_3.py — S2S Device Registry & Whitelist Manager (SECURITY FIXED)

SECURITY FIX (March 19, 2026):
  - Removed verify_signature parameter (was allowing signature bypass)
  - Signature verification is now MANDATORY when public key exists
  - Added audit logging for all verification attempts
  - See git diff for details

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

  # Validate a certificate (signature verification is MANDATORY):
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
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup security audit logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('s2s_registry_security')

# Attempt to import from package, fall back to local
try:
    from .s2s_signing_v1_3 import CertVerifier
except (ImportError, ValueError):
    from s2s_signing_v1_3 import CertVerifier


class DeviceRegistry:
    """
    Device registry and whitelist manager.

    SECURITY NOTE: All certificates with public keys MUST pass signature
    verification. The old verify_signature=False bypass has been removed.
    """

    def __init__(self, registry_path: str = "device_registry.json"):
        self.registry_path = Path(registry_path)
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        """Load registry from JSON file."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.devices = data.get("devices", {})
                logger.info(f"Loaded {len(self.devices)} devices from {self.registry_path}")
        else:
            self.devices = {}
            logger.info(f"Registry file not found, starting fresh: {self.registry_path}")

    def save(self):
        """Save registry to JSON file."""
        data = {
            "version": "1.3.1",  # Incremented for security fix
            "last_updated": time.time(),
            "devices": self.devices,
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.devices)} devices to {self.registry_path}")

    def register(
        self,
        device_id: str,
        sensor_profile: str,
        expected_jitter_ns: float,
        public_key_pem: Optional[str] = None,
        owner: str = "unknown",
        trust_tier: str = "PROVISIONAL",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a new device.

        Args:
            device_id: Unique device identifier
            sensor_profile: Sensor configuration (e.g., "imu_9dof", "emg_8ch")
            expected_jitter_ns: Expected RMS jitter from vendor specs
            public_key_pem: Ed25519 public key for signature verification
            owner: Device owner email/ID
            trust_tier: TRUSTED | PROVISIONAL | UNTRUSTED
            notes: Optional notes

        Returns:
            Device record
        """
        if trust_tier not in ["TRUSTED", "PROVISIONAL", "UNTRUSTED"]:
            raise ValueError(f"Invalid trust_tier: {trust_tier}")

        device = {
            "device_id": device_id,
            "sensor_profile": sensor_profile,
            "expected_jitter_ns": expected_jitter_ns,
            "public_key_pem": public_key_pem,
            "owner": owner,
            "trust_tier": trust_tier,
            "notes": notes,
            "registered_at": time.time(),
            "last_seen": None,
            "total_certs_validated": 0,
            "revoked": False,
        }

        self.devices[device_id] = device
        self.save()
        logger.info(f"Registered device: {device_id} (tier={trust_tier}, owner={owner})")
        return device

    def revoke(self, device_id: str, reason: str):
        """Revoke a device (blacklist)."""
        if device_id not in self.devices:
            raise KeyError(f"Device not found: {device_id}")
        
        self.devices[device_id]["revoked"] = True
        self.devices[device_id]["revoke_reason"] = reason
        self.devices[device_id]["revoked_at"] = time.time()
        self.save()
        logger.warning(f"REVOKED device: {device_id} - Reason: {reason}")

    def validate_cert(
        self,
        cert: Dict[str, Any],
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate a certificate against the registry.

        SECURITY FIX: verify_signature parameter removed. Signature verification
        is now MANDATORY when device has a public key. No bypass possible.

        Checks:
          1. device_id is registered
          2. Device is not revoked
          3. Device trust tier is TRUSTED (warns for PROVISIONAL)
          4. Hardware jitter matches expected spec (if cert has jitter metrics)
          5. Cryptographic signature is valid (if device has a public key) - MANDATORY

        Returns: (is_valid: bool, reason: str, device_record: dict | None)
        """
        device_id = cert.get("device_id") or cert.get("meta_in", {}).get("device_id")

        if not device_id:
            logger.warning("VALIDATION FAILED: Certificate missing device_id")
            return False, "CERT_MISSING_DEVICE_ID", None

        if device_id not in self.devices:
            logger.warning(f"VALIDATION FAILED: Device not registered: {device_id}")
            return False, f"DEVICE_NOT_REGISTERED: {device_id}", None

        device = self.devices[device_id]

        # Check revocation status
        if device.get("revoked", False):
            reason = device.get("revoke_reason", "UNKNOWN")
            logger.warning(f"VALIDATION FAILED: Device revoked: {device_id} ({reason})")
            return False, f"DEVICE_REVOKED: {reason}", device

        # Check trust tier
        tier = device.get("trust_tier", "UNTRUSTED")
        if tier == "UNTRUSTED":
            logger.warning(f"VALIDATION FAILED: Device untrusted: {device_id}")
            return False, "DEVICE_UNTRUSTED", device

        # Hardware jitter validation
        expected = device.get("expected_jitter_ns")
        if expected is not None:
            metrics = cert.get("metrics") or cert.get("meta_in", {}).get("hardware_check", {})
            rms_jitter = metrics.get("rms_jitter_ns")

            if rms_jitter is not None:
                tolerance = 0.30  # ±30% tolerance for hardware variance
                margin = max(expected * tolerance, 1.0)
                if abs(rms_jitter - expected) > margin:
                    logger.warning(
                        f"VALIDATION FAILED: Jitter mismatch for {device_id}: "
                        f"measured={rms_jitter:.1f}ns expected={expected:.1f}ns ±{margin:.1f}ns"
                    )
                    return (
                        False,
                        f"JITTER_MISMATCH: measured={rms_jitter:.1f}ns "
                        f"expected={expected:.1f}ns ±{margin:.1f}ns",
                        device,
                    )

        # SECURITY FIX: Signature verification is now MANDATORY
        # No verify_signature parameter - always verify if public key exists
        if device.get("public_key_pem"):
            sig = cert.get("_signature")
            if not sig:
                logger.error(
                    f"SECURITY: Certificate from {device_id} is unsigned but device has public key"
                )
                return False, "CERT_UNSIGNED_BUT_DEVICE_HAS_KEY", device

            verifier = CertVerifier(
                public_key_bytes=device["public_key_pem"].encode("utf-8")
            )
            ok, sig_reason = verifier.verify_cert(cert)
            if not ok:
                logger.error(
                    f"SECURITY: Signature validation failed for {device_id}: {sig_reason}"
                )
                return False, f"SIGNATURE_INVALID: {sig_reason}", device

            logger.info(f"SECURITY: Signature verified for device {device_id}")

        # Update device stats
        device["last_seen"] = time.time()
        device["total_certs_validated"] = device.get("total_certs_validated", 0) + 1
        self.save()

        # Warn if provisional
        if tier == "PROVISIONAL":
            logger.info(f"VALIDATION PASSED (WARNING: provisional tier): {device_id}")
            return True, "VALID_BUT_PROVISIONAL", device

        logger.info(f"VALIDATION PASSED: {device_id}")
        return True, "VALID", device

    def list_devices(self) -> List[Dict[str, Any]]:
        """List all registered devices."""
        return list(self.devices.values())

    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device record by ID."""
        return self.devices.get(device_id)


def main():
    """CLI for device registry management."""
    parser = argparse.ArgumentParser(description="S2S Device Registry CLI")
    parser.add_argument(
        "--registry",
        default="device_registry.json",
        help="Path to registry file",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register command
    reg_parser = subparsers.add_parser("register", help="Register a new device")
    reg_parser.add_argument("--device-id", required=True, help="Device ID")
    reg_parser.add_argument("--profile", required=True, help="Sensor profile")
    reg_parser.add_argument("--jitter", type=float, required=True, help="Expected jitter (ns)")
    reg_parser.add_argument("--public-key-file", help="Path to public key PEM file")
    reg_parser.add_argument("--owner", default="unknown", help="Device owner")
    reg_parser.add_argument("--tier", default="PROVISIONAL", choices=["TRUSTED", "PROVISIONAL", "UNTRUSTED"])
    reg_parser.add_argument("--notes", help="Optional notes")

    # List command
    subparsers.add_parser("list", help="List all devices")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate a certificate")
    val_parser.add_argument("cert_file", help="Path to certificate JSON file")

    # Revoke command
    rev_parser = subparsers.add_parser("revoke", help="Revoke a device")
    rev_parser.add_argument("device_id", help="Device ID to revoke")
    rev_parser.add_argument("--reason", required=True, help="Revocation reason")

    args = parser.parse_args()

    registry = DeviceRegistry(args.registry)

    if args.command == "register":
        public_key_pem = None
        if args.public_key_file:
            with open(args.public_key_file, "r") as f:
                public_key_pem = f.read()

        device = registry.register(
            device_id=args.device_id,
            sensor_profile=args.profile,
            expected_jitter_ns=args.jitter,
            public_key_pem=public_key_pem,
            owner=args.owner,
            trust_tier=args.tier,
            notes=args.notes,
        )
        print(f"✅ Registered: {device['device_id']}")
        print(json.dumps(device, indent=2))

    elif args.command == "list":
        devices = registry.list_devices()
        print(f"\n📋 {len(devices)} devices registered:\n")
        for d in devices:
            status = "🔴 REVOKED" if d.get("revoked") else "✅ ACTIVE"
            print(f"{status} {d['device_id']:20} | {d['trust_tier']:12} | Owner: {d['owner']}")

    elif args.command == "validate":
        with open(args.cert_file, "r") as f:
            cert = json.load(f)
        
        # SECURITY: Note that verify_signature parameter has been removed
        # Signature verification is now mandatory
        ok, reason, device = registry.validate_cert(cert)
        
        if ok:
            print(f"✅ VALID: {reason}")
            if device:
                print(f"   Device: {device['device_id']}")
                print(f"   Tier: {device['trust_tier']}")
                print(f"   Owner: {device['owner']}")
        else:
            print(f"❌ INVALID: {reason}")
            if device:
                print(f"   Device: {device.get('device_id', 'unknown')}")

    elif args.command == "revoke":
        registry.revoke(args.device_id, args.reason)
        print(f"🔴 Revoked: {args.device_id}")
        print(f"   Reason: {args.reason}")


if __name__ == "__main__":
    main()
