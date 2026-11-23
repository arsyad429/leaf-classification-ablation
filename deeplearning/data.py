"""Prepare the same photo inputs and image treatments as the ML benchmark."""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "deeplearning" / "splits"
SEED = 42
SIZE = 224
CONDITIONS = ("N", "E", "S-K")
AUGMENTATIONS = ("noise", "flip", "rotation", "corruption", "brightness", "contrast")


def read_csv(name):
    with (SPLITS / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def load_photo(row):
    with Image.open(ROOT / row["path"]) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    if int(row["rotation_ccw_degrees"]):
        photo = photo.rotate(int(row["rotation_ccw_degrees"]), expand=True)
    scale = min(1.0, 640 / max(photo.size))
    size = tuple(max(1, round(dimension * scale)) for dimension in photo.size)
    return np.asarray(photo.resize(size, Image.Resampling.LANCZOS))


def augment(rgb, operation, rng):
    image = rgb.astype(np.float32)
    if operation == "noise":
        image += rng.normal(0, rng.uniform(3.0, 10.0), image.shape).astype(np.float32)
    elif operation == "flip":
        image = np.flip(image, axis=int(rng.integers(0, 2)))
    elif operation == "rotation":
        image = np.rot90(image, k=int(rng.integers(1, 4)))
    elif operation == "corruption":
        fraction = rng.uniform(0.002, 0.008)
        choices = rng.random(image.shape[:2])
        image[choices < fraction / 2] = 0
        image[(choices >= fraction / 2) & (choices < fraction)] = 255
    elif operation == "brightness":
        image += rng.uniform(10.0, 20.0) * rng.choice([-1.0, 1.0])
    elif operation == "contrast":
        center = image.mean(axis=(0, 1), keepdims=True)
        image = (image - center) * rng.uniform(0.85, 1.15) + center
    return np.ascontiguousarray(np.clip(np.rint(image), 0, 255).astype(np.uint8))


def preprocess(rgb, condition):
    if condition == "E":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if condition == "S-K":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        rng = np.random.default_rng(SEED)
        sample = lab[rng.choice(len(lab), min(len(lab), 5000), replace=False)]
        model = MiniBatchKMeans(n_clusters=4, random_state=SEED, n_init=3, batch_size=512)
        model.fit(sample)
        foreground = np.flatnonzero(model.cluster_centers_[:, 1] < 128)
        if not foreground.size:
            foreground = [model.cluster_centers_[:, 1].argmin()]
        mask = np.isin(model.predict(lab), foreground).reshape(rgb.shape[:2]).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return np.where(mask[:, :, None], rgb, 0)
    return rgb


def resize_for_model(rgb):
    height, width = rgb.shape[:2]
    scale = SIZE / max(height, width)
    size = tuple(round(value * scale) for value in (width, height))
    resized = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE))
    canvas.paste(resized, ((SIZE - size[0]) // 2, (SIZE - size[1]) // 2))
    return np.asarray(canvas).transpose(2, 0, 1)


class BenchmarkData:
    def __init__(self):
        rows = read_csv("images.csv")
        self.ids = [row["image"] for row in rows]
        self.classes = sorted({row["label"] for row in rows})
        self.labels = torch.tensor([self.classes.index(row["label"]) for row in rows])
        indices = {image: i for i, image in enumerate(self.ids)}
        self.outer = {}
        self.inner = {}
        for row in read_csv("outer.csv"):
            key = (int(row["fold"]), row["role"])
            self.outer.setdefault(key, []).append(indices[row["image"]])
        for row in read_csv("inner.csv"):
            key = (int(row["outer_fold"]), int(row["inner_fold"]), row["role"])
            self.inner.setdefault(key, []).append(indices[row["image"]])
        self.photos = [load_photo(row) for row in rows]

    def input_bank(self, condition, device):
        originals = np.empty((len(self.photos), 3, SIZE, SIZE), dtype=np.uint8)
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            for i, original in enumerate(self.photos):
                originals[i] = resize_for_model(preprocess(original, condition))
        return InputBank(condition, self.photos, torch.from_numpy(originals).to(device))


class InputBank:
    def __init__(self, condition, photos, originals):
        self.condition = condition
        self.photos = photos
        self.originals = originals
        self.variants = {}

    def training_variants(self, indices):
        # Only construct a variant through an explicit training subset.
        with threadpool_limits(limits=1):
            for i in indices:
                if i in self.variants:
                    continue
                variants = np.empty((len(AUGMENTATIONS), 3, SIZE, SIZE), dtype=np.uint8)
                for version, operation in enumerate(AUGMENTATIONS, 1):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, i, version, 0]))
                    augmented = augment(self.photos[i], operation, rng)
                    variants[version - 1] = resize_for_model(preprocess(augmented, self.condition))
                self.variants[i] = torch.from_numpy(variants).to(self.originals.device)
        return torch.stack([
            torch.cat((self.originals[i:i + 1], self.variants[i])) for i in indices
        ])
