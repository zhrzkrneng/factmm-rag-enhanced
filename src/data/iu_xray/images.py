"""IU X-Ray lazy, CPU-only image loading.

Responsibility: open and validate the readability of IU X-Ray image
files on demand -- never at record-construction or dataset-load time.
No resizing, normalization, or model-specific preprocessing here; this
is a basic loader smoke-test layer only, matching this cell's explicit
"no model-specific preprocessing yet" scope. Follows the same lazy-PIL-
import-inside-the-function convention already established by
src/baseline/generation/adapter.py's HFGeneratorAdapter.prepare_inputs,
for consistency, though PIL/torch are not GPU-bound here and no torch
import happens at all in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.data.iu_xray.exceptions import IuXrayPathResolutionError


def open_image(path: str):
    """Opens and returns one image, converted to RGB, CPU only.

    Args:
        path: an already-resolved (non-token) filesystem path.

    Returns:
        A PIL.Image.Image in RGB mode.

    Raises:
        IuXrayPathResolutionError: if the file does not exist, or exists
            but cannot be read as an image (corrupt/unsupported file).
    """
    if not Path(path).exists():
        raise IuXrayPathResolutionError(
            f"Image file not found: {path!r}"
        )
    from PIL import Image, UnidentifiedImageError

    try:
        return Image.open(path).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise IuXrayPathResolutionError(
            f"Failed to read image at {path!r} (corrupt or unsupported "
            f"file): {exc}"
        ) from exc


def validate_readable(path: str) -> None:
    """Raises IuXrayPathResolutionError iff `path` is not a readable
    image; returns None (does not return the opened image) so callers
    that only need a smoke-test check don't hold image data in memory.

    open_image() already forces a full decode via .convert("RGB"), which
    is sufficient to surface corruption -- no separate PIL .verify() call
    is made here, since .verify() is only meaningful on a freshly opened,
    not-yet-decoded image and would be redundant (or spuriously wrong)
    after convert() has already succeeded.
    """
    open_image(path)


def load_images(paths: List[str]):
    """Opens every path in `paths`, in order, lazily (one at a time).

    Raises:
        IuXrayPathResolutionError: on the first unreadable path, naming
            exactly which one and why -- never silently skips a missing
            or corrupt image out of a multi-image study.
    """
    return [open_image(path) for path in paths]
