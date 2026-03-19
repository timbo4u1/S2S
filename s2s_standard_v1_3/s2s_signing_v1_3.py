#!/usr/bin/env python3
"""
s2s_signing_v1_3.py — Ed25519 Certificate Signing for S2S v1.3

Provides cryptographic signing and verification of S2S JSON certificates.
Uses Ed25519 (fast, compact 64-byte signatures, no external dependencies
beyond the standard `cryptography` package).

If `cryptography` is not installed, falls back to a HMAC-SHA256 mode
using only stdlib — still provides tamper-detection, just not asymmetric.

Why sign certificates?
  - Proves a specific S2S node produced the certificate (not a third party)
  - Marketplace buyers can verify cert authenticity before paying
  - Ancestry royalties require traceable cert lineage
  - Prevents cert forgery / replay attacks

Key management:
  # Generate a keypair (do this once per device):
  python3 -m s2s_standard_v1_3.s2s_signing_v1_3 keygen --out keys/device_001

  # This creates:
  #   keys/device_001.private.pem  — keep secret, never share
  #   keys/device_001.public.pem   — share with marketplace / verifiers

Usage in code:
  from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner, CertVerifier

  # Sign a certificate dict:
  signer = CertSigner.from_pem_file("keys/device_001.private.pem")
  signed_cert = signer.sign_cert(cert_dict)
  # signed_cert now has: cert["_signature"] and cert["_signing_key_id"]

  # Verify:
  verifier = CertVerifier.from_pem_file("keys/device_001.public.pem")
  ok, reason = verifier.verify_cert(signed_cert)

Usage via StreamCertifier (automatic signing):
  from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier
  from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner

  signer = CertSigner.from_pem_file("keys/device_001.private.pem")
  sc = StreamCertifier(sensor_names=[...], signer=signer)
  # Every emitted cert is automatically signed
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Try to import Ed25519 from cryptography package
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat, PrivateFormat, NoEncryption,
        load_pem_private_key, load_pem_public_key,
    )
    from cryptography.exceptions import InvalidSignature
    ED25519_AVAILABLE = True
except ImportError:
    ED25519_AVAILABLE = False

SIGNING_MODE_ED25519 = "Ed25519"
SIGNING_MODE_HMAC    = "HMAC-SHA256"   # stdlib fallback

# Fields excluded from the signed payload (they don't exist at signing time
# or are the signature itself)
_EXCLUDED_FIELDS = {"_signature", "_signing_key_id", "_signing_mode", "_signed_at_ns"}


# ---------------------------------------------------------------------------
# Canonical payload serialization
# ---------------------------------------------------------------------------

def _canonical_payload(cert: Dict[str, Any]) -> bytes:
    """
    Produce a deterministic bytes representation of the cert for signing.
    Excludes signature fields. Uses compact JSON with sorted keys.
    """
    filtered = {k: v for k, v in cert.items() if k not in _EXCLUDED_FIELDS}
    return json.dumps(filtered, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Key ID derivation
# ---------------------------------------------------------------------------

def _key_id_from_public_bytes(pub_bytes: bytes) -> str:
    """Short hex fingerprint of a public key (first 8 bytes of SHA256)."""
    return hashlib.sha256(pub_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CertSigner
# ---------------------------------------------------------------------------

class CertSigner:
    """
    Signs S2S certificate dicts using Ed25519 (or HMAC-SHA256 fallback).

    Attaches these fields to the cert:
      _signature     : base64url-encoded signature bytes
      _signing_key_id: short hex fingerprint of the public key
      _signing_mode  : "Ed25519" or "HMAC-SHA256"
      _signed_at_ns  : signing timestamp (nanoseconds)
    """

    def __init__(
        self,
        private_key_bytes: Optional[bytes] = None,
        hmac_secret: Optional[bytes] = None,
        key_id: Optional[str] = None,
    ):
        """
        Prefer Ed25519 if private_key_bytes provided and cryptography available.
        Falls back to HMAC-SHA256 if hmac_secret provided.
        """
        self._ed25519_key = None
        self._hmac_secret = None
        self._key_id      = key_id or "unknown"
        self._mode        = SIGNING_MODE_HMAC

        if private_key_bytes and ED25519_AVAILABLE:
            try:
                self._ed25519_key = load_pem_private_key(private_key_bytes, password=None)
                pub_bytes = self._ed25519_key.public_key().public_bytes(
                    Encoding.Raw, PublicFormat.Raw
                )
                self._key_id = _key_id_from_public_bytes(pub_bytes)
                self._mode   = SIGNING_MODE_ED25519
            except Exception:
                # Try raw bytes (32-byte private seed)
                if len(private_key_bytes) == 32:
                    self._ed25519_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
                    pub_bytes = self._ed25519_key.public_key().public_bytes(
                        Encoding.Raw, PublicFormat.Raw
                    )
                    self._key_id = _key_id_from_public_bytes(pub_bytes)
                    self._mode   = SIGNING_MODE_ED25519

        if self._ed25519_key is None:
            # Fallback: HMAC-SHA256
            self._hmac_secret = hmac_secret or os.urandom(32)
            if not key_id:
                self._key_id = hashlib.sha256(self._hmac_secret).hexdigest()[:16]
            self._mode = SIGNING_MODE_HMAC

    @classmethod
    def from_pem_file(cls, path: str) -> "CertSigner":
        return cls(private_key_bytes=Path(path).read_bytes())

    @classmethod
    def from_hmac_secret(cls, secret: bytes, key_id: Optional[str] = None) -> "CertSigner":
        return cls(hmac_secret=secret, key_id=key_id)

    @classmethod
    def generate(cls) -> Tuple["CertSigner", "CertVerifier"]:
        """
        Generate a fresh Ed25519 keypair and return (signer, verifier).
        Use this for ephemeral keys (e.g. unit tests, demos).
        """
        if not ED25519_AVAILABLE:
            # fallback: shared HMAC secret
            secret = os.urandom(32)
            signer   = cls.from_hmac_secret(secret)
            verifier = CertVerifier.from_hmac_secret(secret, key_id=signer.key_id)
            return signer, verifier

        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        public_pem  = private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        signer   = cls(private_key_bytes=private_pem)
        verifier = CertVerifier(public_key_bytes=public_pem)
        return signer, verifier

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def mode(self) -> str:
        return self._mode

    def sign_cert(self, cert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a new dict with signature fields added.
        Does NOT modify the original cert.
        """
        signed = dict(cert)
        signed["_signing_key_id"] = self._key_id
        signed["_signing_mode"]   = self._mode
        signed["_signed_at_ns"]   = time.time_ns()

        payload = _canonical_payload(signed)

        if self._mode == SIGNING_MODE_ED25519 and self._ed25519_key:
            sig_bytes = self._ed25519_key.sign(payload)
        else:
            sig_bytes = hmac.new(self._hmac_secret, payload, hashlib.sha256).digest()

        signed["_signature"] = base64.urlsafe_b64encode(sig_bytes).decode("ascii")
        return signed

    def export_public_pem(self) -> Optional[str]:
        """Return PEM-encoded public key string, or None if HMAC mode."""
        if self._ed25519_key and ED25519_AVAILABLE:
            return self._ed25519_key.public_key().public_bytes(
                Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
            ).decode("ascii")
        return None

    def save_keypair(self, base_path: str) -> Tuple[str, Optional[str]]:
        """
        Save private key to <base_path>.private.pem and public key to
        <base_path>.public.pem. Returns (private_path, public_path).
        """
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)

        if self._ed25519_key and ED25519_AVAILABLE:
            priv_pem = self._ed25519_key.private_bytes(
                Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
            )
            pub_pem = self._ed25519_key.public_key().public_bytes(
                Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
            )
            priv_path = str(base) + ".private.pem"
            pub_path  = str(base) + ".public.pem"
            Path(priv_path).write_bytes(priv_pem)
            Path(pub_path).write_bytes(pub_pem)
            return priv_path, pub_path
        else:
            # Save HMAC secret
            secret_path = str(base) + ".hmac.secret"
            Path(secret_path).write_bytes(self._hmac_secret)
            return secret_path, None


# ---------------------------------------------------------------------------
# CertVerifier
# ---------------------------------------------------------------------------

class CertVerifier:
    """
    Verifies signatures on S2S certificate dicts.
    """

    def __init__(
        self,
        public_key_bytes: Optional[bytes] = None,
        hmac_secret: Optional[bytes] = None,
        key_id: Optional[str] = None,
    ):
        self._ed25519_pub = None
        self._hmac_secret = None
        self._key_id      = key_id or "unknown"
        self._mode        = SIGNING_MODE_HMAC

        if public_key_bytes and ED25519_AVAILABLE:
            try:
                self._ed25519_pub = load_pem_public_key(public_key_bytes)
                raw = self._ed25519_pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
                self._key_id = _key_id_from_public_bytes(raw)
                self._mode   = SIGNING_MODE_ED25519
            except Exception:
                pass

        if self._ed25519_pub is None and hmac_secret:
            self._hmac_secret = hmac_secret
            if not key_id:
                self._key_id = hashlib.sha256(hmac_secret).hexdigest()[:16]
            self._mode = SIGNING_MODE_HMAC

    @classmethod
    def from_pem_file(cls, path: str) -> "CertVerifier":
        return cls(public_key_bytes=Path(path).read_bytes())

    @classmethod
    def from_hmac_secret(cls, secret: bytes, key_id: Optional[str] = None) -> "CertVerifier":
        return cls(hmac_secret=secret, key_id=key_id)

    @property
    def key_id(self) -> str:
        return self._key_id

    def verify_cert(self, cert: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify a signed cert dict.
        Returns (is_valid: bool, reason: str).
        """
        sig_b64  = cert.get("_signature")
        key_id   = cert.get("_signing_key_id")
        mode     = cert.get("_signing_mode")

        if not sig_b64:
            return False, "NO_SIGNATURE_FIELD"

        try:
            sig_bytes = base64.urlsafe_b64decode(sig_b64 + "==")
        except Exception:
            return False, "INVALID_SIGNATURE_ENCODING"

        payload = _canonical_payload(cert)

        if mode == SIGNING_MODE_ED25519 and self._ed25519_pub and ED25519_AVAILABLE:
            try:
                self._ed25519_pub.verify(sig_bytes, payload)
                return True, "VALID_ED25519"
            except Exception:
                return False, "INVALID_ED25519_SIGNATURE"

        elif self._hmac_secret:
            expected = hmac.new(self._hmac_secret, payload, hashlib.sha256).digest()
            if hmac.compare_digest(sig_bytes, expected):
                return True, "VALID_HMAC_SHA256"
            return False, "INVALID_HMAC_SIGNATURE"

        return False, "NO_VERIFICATION_KEY_AVAILABLE"


# ---------------------------------------------------------------------------
# StreamCertifier integration helper
# ---------------------------------------------------------------------------

def attach_signer_to_certifier(certifier: Any, signer: "CertSigner") -> None:
    """
    Monkey-patch a StreamCertifier (or any certifier with _evaluate_window)
    to auto-sign every emitted certificate.

    Usage:
        sc = StreamCertifier(sensor_names=[...])
        signer = CertSigner.from_pem_file("device.private.pem")
        attach_signer_to_certifier(sc, signer)
        # Now every cert from sc.push_frame() is automatically signed
    """
    original_evaluate = certifier._evaluate_window

    def _signed_evaluate():
        cert = original_evaluate()
        return signer.sign_cert(cert)

    certifier._evaluate_window = _signed_evaluate


# ---------------------------------------------------------------------------
# CLI — key generation + cert verification
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S2S v1.3 — Certificate Signing Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # keygen
    kg = sub.add_parser("keygen", help="Generate a new Ed25519 keypair")
    kg.add_argument("--out", required=True,
                    help="Base path for keypair (e.g. keys/device_001)")

    # verify
    vf = sub.add_parser("verify", help="Verify a signed certificate JSON file")
    vf.add_argument("cert_file", help="Path to signed .json certificate")
    vf.add_argument("--pubkey", required=True, help="Path to .public.pem file")

    # sign
    sg = sub.add_parser("sign", help="Sign a certificate JSON file")
    sg.add_argument("cert_file", help="Path to .json certificate to sign")
    sg.add_argument("--privkey", required=True, help="Path to .private.pem file")
    sg.add_argument("--out", default=None, help="Output path (default: overwrite)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "keygen":
        signer, _ = CertSigner.generate()
        priv_path, pub_path = signer.save_keypair(args.out)
        print(json.dumps({
            "event":       "keypair_generated",
            "mode":        signer.mode,
            "key_id":      signer.key_id,
            "private_key": priv_path,
            "public_key":  pub_path,
        }, indent=2))

    elif args.command == "verify":
        cert = json.loads(Path(args.cert_file).read_text())
        verifier = CertVerifier.from_pem_file(args.pubkey)
        ok, reason = verifier.verify_cert(cert)
        print(json.dumps({
            "file":    args.cert_file,
            "valid":   ok,
            "reason":  reason,
            "key_id":  verifier.key_id,
        }, indent=2))

    elif args.command == "sign":
        cert    = json.loads(Path(args.cert_file).read_text())
        signer  = CertSigner.from_pem_file(args.privkey)
        signed  = signer.sign_cert(cert)
        out_path = args.out or args.cert_file
        Path(out_path).write_text(json.dumps(signed, indent=2, default=str))
        print(json.dumps({
            "event":   "cert_signed",
            "output":  out_path,
            "key_id":  signer.key_id,
            "mode":    signer.mode,
        }, indent=2))


if __name__ == "__main__":
    main()
