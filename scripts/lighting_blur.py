"""Controlled test-image perturbations for the lighting and blur study."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter

BRIGHTNESS = {"brightness_085": 0.85, "brightness_070": 0.70}
BLUR = {"blur_075": 0.75, "blur_150": 1.50}
CONDITIONS = ("original", *BRIGHTNESS, *BLUR)


def perturb(image, condition):
    if condition == "original":
        return image
    if condition in BRIGHTNESS:
        pixels = np.asarray(image, dtype=np.float64)
        return Image.fromarray(np.rint(pixels * BRIGHTNESS[condition]).astype(np.uint8))
    return image.filter(ImageFilter.GaussianBlur(radius=BLUR[condition]))


@contextmanager
def study_data(data, category, condition, resize):
    import cv2

    with TemporaryDirectory(prefix="mvtec_test_") as folder:
        root = Path(folder)
        target = root / category
        target.mkdir()
        (target / "train").symlink_to(data / category / "train", target_is_directory=True)
        for source in sorted((data / category / "test").glob("*/*.png")):
            output = target / source.relative_to(data / category)
            output.parent.mkdir(exist_ok=True, parents=True)
            rgb = cv2.cvtColor(cv2.imread(str(source), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            resized = resize(Image.fromarray(rgb))
            perturb(resized, condition).save(output)
        yield root
