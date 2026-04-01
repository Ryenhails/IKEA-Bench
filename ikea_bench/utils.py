"""Shared utility functions for IKEA-Bench evaluation."""

import re
from pathlib import Path

# Lazy imports for torch/PIL -- only needed when running actual models
torch = None
Image = None


def _ensure_imports():
    """Lazily import torch and PIL so lightweight CLI usage stays fast."""
    global torch, Image
    if torch is None:
        import torch as _torch
        torch = _torch
    if Image is None:
        from PIL import Image as _Image
        Image = _Image


def get_project_root() -> Path:
    """Return the root directory of the IKEA-Bench project."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the default data directory (<project_root>/data)."""
    return get_project_root() / "data"


def extract_answer(response: str, valid_labels: list) -> str | None:
    """Extract answer letter from model response.

    Expects answer-first format: 'Answer: X. Reason: ...'
    Strips thinking blocks (GLM-4.1V-Thinking, Qwen3.5 thinking mode, etc.).
    """
    # Strip thinking blocks -- handle both closed and unclosed (truncated)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()

    # Priority 1: "Answer: X" -- extract first letter after "Answer"
    m = re.search(r'[Aa]nswer[:\s]*([A-Da-d])', response)
    if m:
        letter = m.group(1).upper()
        if letter in valid_labels:
            return letter

    # Priority 2: First letter A-D in response (answer comes first in our prompt)
    m = re.search(r'\b([A-Da-d])\b', response)
    if m:
        letter = m.group(1).upper()
        if letter in valid_labels:
            return letter

    # Priority 3: Direct single letter
    cleaned = response.strip().upper()
    if cleaned in valid_labels:
        return cleaned

    return None  # Could not parse
