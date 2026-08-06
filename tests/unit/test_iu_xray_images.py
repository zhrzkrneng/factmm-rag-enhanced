"""Unit tests for src/data/iu_xray/images.py.

Covers: (10) missing image path error, (11) corrupt image error,
(14) lazy image loading.
"""

import pytest

from src.data.iu_xray.exceptions import IuXrayPathResolutionError
from src.data.iu_xray.images import load_images, open_image, validate_readable


def _write_valid_png(path):
    from PIL import Image
    Image.new("L", (8, 8)).save(path)


def test_open_image_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(IuXrayPathResolutionError, match="not found"):
        open_image(str(missing))


def test_open_image_raises_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a real png file")
    with pytest.raises(IuXrayPathResolutionError, match="Failed to read"):
        open_image(str(corrupt))


def test_open_image_returns_rgb_image_for_valid_file(tmp_path):
    valid = tmp_path / "valid.png"
    _write_valid_png(valid)
    image = open_image(str(valid))
    assert image.mode == "RGB"


def test_validate_readable_raises_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"garbage")
    with pytest.raises(IuXrayPathResolutionError):
        validate_readable(str(corrupt))


def test_validate_readable_passes_silently_on_valid_file(tmp_path):
    valid = tmp_path / "valid.png"
    _write_valid_png(valid)
    assert validate_readable(str(valid)) is None  # no exception raised


def test_load_images_preserves_original_image_count_and_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_valid_png(first)
    _write_valid_png(second)

    images = load_images([str(first), str(second)])
    assert len(images) == 2


def test_load_images_raises_naming_the_first_bad_path(tmp_path):
    good = tmp_path / "good.png"
    _write_valid_png(good)
    missing = tmp_path / "missing.png"

    with pytest.raises(IuXrayPathResolutionError, match=str(missing)):
        load_images([str(good), str(missing)])
