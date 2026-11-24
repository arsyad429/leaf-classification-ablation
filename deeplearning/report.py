"""Export paired comparisons with scratch training and the classical SVM."""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
import time


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "deeplearning" / "results" / "benchmark"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def classical_runs():
    baselines = {}
    for augmentation, folder in ((False, "hasil_ml_full"), (True, "hasil_ml_full_aug")):
        results = ROOT / "ablasi ML awal" / folder
        match = dict(input="N", feature="F4", model="SVM")
        folds = {}
        for row in read_csv(results / "metrics_per_fold.csv"):
            if all(row[key] == value for key, value in match.items()):
                folds[int(row["fold"])] = dict(macro_f1=float(row["macro_f1"]), predictions={})
        for row in read_csv(results / "oof_predictions.csv"):
            if all(row[key] == value for key, value in match.items()):
                folds[int(row["fold"])]["predictions"][row["image"]] = (row["true"], row["pred"])
        baselines[augmentation] = folds
    return baselines


def deep_learning_runs(journal):
    runs = {}
    if not journal.exists():
        return runs
    for line in journal.read_text(encoding="utf-8").rpartition("\n")[0].splitlines():
        row = json.loads(line)
        config = row["config"]
        key = tuple(config[name] for name in ("input", "architecture", "initialization", "augmentation"))
        predictions = {item["image"]: (item["true"], item["pred"]) for item in row["predictions"]}
        runs.setdefault(key, {})[row["fold"]] = dict(macro_f1=row["macro_f1"], predictions=predictions)
    return runs


def compare(left, right):
    if set(left) != set(right) or len(left) != 5:
        raise ValueError("A paired comparison needs matching outer folds")
    left_only, right_only, photos = 0, 0, 0
    for fold, first in sorted(left.items()):
        second = right[fold]
        a, b = first["predictions"], second["predictions"]
        if set(a) != set(b):
            raise ValueError(f"Different evaluation photos on fold {fold}")
        for image, (label, prediction) in a.items():
            other_label, other_prediction = b[image]
            if label != other_label:
                raise ValueError(f"Class label mismatch for {image}")
            first_correct = prediction == label
            second_correct = other_prediction == label
            left_only += first_correct and not second_correct
            right_only += second_correct and not first_correct
            photos += 1
    macro_left = mean(row["macro_f1"] for row in left.values())
    macro_right = mean(row["macro_f1"] for row in right.values())
    return dict(photos=photos, macro_f1_left=macro_left, macro_f1_right=macro_right,
                macro_f1_delta=macro_left - macro_right,
                only_left_correct=left_only, only_right_correct=right_only)


def comparisons(runs, baselines):
    rows = []
    for (condition, architecture, initialization, aug), folds in sorted(runs.items()):
        if len(folds) != 5:
            continue
        left = f"{architecture}/{initialization}"
        if condition == "N":
            rows.append(dict(input=condition, augmentation=aug, left=left, right="F4/SVM",
                             **compare(folds, baselines[aug])))
        if initialization == "pretrained":
            scratch = runs.get((condition, architecture, "scratch", aug), {})
            if len(scratch) == 5:
                rows.append(dict(input=condition, augmentation=aug, left=left,
                                 right=f"{architecture}/scratch", **compare(folds, scratch)))
    return rows


def publish(path, rows):
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Update paired comparisons as configurations finish")
    args = parser.parse_args()
    baselines = classical_runs()
    previous_count = 0
    while True:
        runs = deep_learning_runs(BENCHMARK / "folds.jsonl")
        rows = comparisons(runs, baselines)
        if len(rows) > previous_count:
            publish(BENCHMARK / "comparisons.csv", rows)
            previous_count = len(rows)
        if not args.watch or len(rows) == 30:
            print(f"Paired comparisons: {len(rows)}")
            return
        log_path = BENCHMARK / "run.log"
        lines = log_path.read_text(errors="replace").rstrip().splitlines() if log_path.exists() else []
        if lines and (lines[-1] == "Benchmark complete" or "subprocess.CalledProcessError:" in lines[-1]):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
