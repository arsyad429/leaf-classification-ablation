"""Rebuild the published ML folds for the deep learning benchmark."""

import argparse
import csv
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold


PROJECT = Path(__file__).resolve().parents[1]
SEED = 42


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pixel_hash(rgb):
    shape = (rgb.height, rgb.width, 3)
    return hashlib.sha256(str(shape).encode() + rgb.tobytes()).hexdigest()


def rotation_to_match(rgb, expected):
    if pixel_hash(rgb) == expected:
        return 0
    for degrees, transpose in (
        (90, Image.Transpose.ROTATE_90),
        (180, Image.Transpose.ROTATE_180),
        (270, Image.Transpose.ROTATE_270),
    ):
        if pixel_hash(rgb.transpose(transpose)) == expected:
            return degrees
    return None


def image_inventory(dataset, audit):
    metadata = read_csv(dataset / "metadata.csv")
    if any("plant_id" in row for row in metadata):
        raise ValueError("The published benchmark uses photo-level splits without plant_id")
    if [(row["image"], row["label"]) for row in metadata] != [
        (row["image"], row["label"]) for row in audit
    ]:
        raise ValueError("Dataset image names, labels, or row order differ from the ML audit")

    inventory = io.StringIO()
    writer = csv.writer(inventory, lineterminator="\n")
    writer.writerow(("image", "label", "path", "rotation_ccw_degrees", "pixel_hash"))
    seen_images = set()
    seen_hashes = set()
    for row in audit:
        image, label, expected = row["image"], row["label"], row["pixel_hash"]
        image_path = dataset / label / image
        with Image.open(image_path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            degrees = rotation_to_match(rgb, expected)
        if degrees is None:
            raise ValueError(f"Image pixels differ from the ML audit after rotation: {image}")
        if image in seen_images or expected in seen_hashes:
            raise ValueError(f"Duplicate image or pixels in the photo-level benchmark: {image}")
        seen_images.add(image)
        seen_hashes.add(expected)
        writer.writerow((image, label, image_path.relative_to(PROJECT).as_posix(), degrees, expected))
    return inventory.getvalue().encode("utf-8")


def partition(indices, labels, folds, seed):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    subset_labels = [labels[index] for index in indices]
    expected_labels = set(labels)
    for train, evaluation in splitter.split(indices, subset_labels):
        train_indices = [indices[index] for index in train]
        evaluation_indices = [indices[index] for index in evaluation]
        if {labels[index] for index in train_indices} != expected_labels or {
            labels[index] for index in evaluation_indices
        } != expected_labels:
            raise ValueError("A fold does not contain all classes")
        yield train_indices, evaluation_indices


def split_files():
    baseline = PROJECT / "ablasi ML awal" / "hasil_ml_full"
    augmented = PROJECT / "ablasi ML awal" / "hasil_ml_full_aug"
    outer_source = baseline / "splits.csv"
    outer_bytes = outer_source.read_bytes()
    if outer_bytes != (augmented / "splits.csv").read_bytes():
        raise ValueError("Published baseline and augmentation splits differ")
    audit = read_csv(baseline / "manifest_audit.csv")
    images = image_inventory(PROJECT / "Dataset", audit)
    labels = [row["label"] for row in audit]

    outer_rows = []
    inner = io.StringIO()
    writer = csv.writer(inner, lineterminator="\n")
    writer.writerow(("outer_fold", "inner_fold", "image", "role", "group"))

    for fold, (train, evaluation) in enumerate(partition(list(range(len(audit))), labels, 5, SEED)):
        for role, indices in (("train", train), ("evaluation", evaluation)):
            outer_rows.extend(
                dict(fold=str(fold), image=audit[index]["image"], role=role, group=audit[index]["pixel_hash"])
                for index in indices
            )
        # Match notebook split_indices on outer-train in metadata order.
        for inner_fold, (inner_train, validation) in enumerate(partition(train, labels, 3, SEED + fold + 1)):
            for role, indices in (("train", inner_train), ("validation", validation)):
                writer.writerows(
                    (fold, inner_fold, audit[index]["image"], role, audit[index]["pixel_hash"])
                    for index in indices
                )

    if outer_rows != read_csv(outer_source):
        raise ValueError("Reconstructed outer memberships differ from the published ML split")
    return {"outer.csv": outer_bytes, "inner.csv": inner.getvalue().encode("utf-8"), "images.csv": images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the saved files without rewriting them")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "splits"
    files = split_files()
    if not args.check:
        output.mkdir(exist_ok=True)
    for name, content in files.items():
        path = output / name
        if args.check:
            if path.read_bytes() != content:
                raise ValueError(f"Saved manifest differs from the ML split: {path}")
        else:
            path.write_bytes(content)
        print(f"{'Verified' if args.check else 'Wrote'} {path}")


if __name__ == "__main__":
    main()
