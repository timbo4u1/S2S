# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.4.x   | ✅ Yes    |
| < 1.4   | ❌ No     |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email: **s2s.physical@proton.me**  
Subject line: `[S2S SECURITY] brief description`

You will receive a response within 48 hours.

## What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Scope

- Physics certification logic (incorrect PASS on invalid data)
- Ed25519 signing/verification bypass
- CLI input handling (path traversal, injection)
- Supply chain (dependency tampering)

## Out of Scope

- Issues in optional dependencies (numpy, torch, streamlit)
- Theoretical attacks without proof of concept

## Disclosure Policy

We follow coordinated disclosure. Once a fix is released, we will publicly acknowledge the reporter (unless they prefer anonymity).
