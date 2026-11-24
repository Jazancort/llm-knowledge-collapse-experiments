# Security Policy

## Scope

This repository hosts experimental training, evaluation, and plotting scripts for the research study on recursive knowledge degradation in LLMs (`pyproject.toml`, `scripts/`, `src/`, and `configs/`).

## Reporting a Vulnerability or Credential Exposure

If you discover a security vulnerability, unsafe deserialization path, or accidental credential exposure in any branch or artifact:

1. Do not open a public issue containing secrets or exploit payloads.
2. Report it privately through GitHub's **Security Advisories** tab (`Report a vulnerability`) on this repository.
3. For non-sensitive dependency updates or environment issues, open a regular GitHub Issue with reproduction details.

## Maintainer Hygiene

- Never commit `.env` files, Hugging Face tokens, or API credentials.
- Dependencies are pinned or bounded in `pyproject.toml` and `requirements.txt`; review third-party model weights and remote code flags (`trust_remote_code`) before running external checkpoints.
