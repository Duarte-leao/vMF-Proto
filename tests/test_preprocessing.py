import hashlib
import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_cropper_preserves_original_images(tmp_path):
    raw = tmp_path / "CUB_200_2011"
    folder = raw / "images" / "001.bird"
    folder.mkdir(parents=True)
    original = folder / "bird.jpg"
    Image.new("RGB", (32, 24), color="green").save(original)
    digest = hashlib.sha256(original.read_bytes()).digest()
    (raw / "images.txt").write_text("1 001.bird/bird.jpg\n")
    (raw / "bounding_boxes.txt").write_text("1 4 5 16 12\n")
    (raw / "train_test_split.txt").write_text("1 1\n")
    (raw / "classes.txt").write_text("1 001.bird\n")
    script = Path(__file__).resolve().parents[1] / "scripts/preprocess/cropimages.py"
    subprocess.run(
        [sys.executable, str(script), "--data-root", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    assert hashlib.sha256(original.read_bytes()).digest() == digest
    with Image.open(
        tmp_path / "cub200_cropped/train_cropped/001.bird/bird.jpg"
    ) as cropped:
        assert cropped.size == (16, 12)
