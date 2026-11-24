"""Train on all primary photos and evaluate on the untouched secondary set."""

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import torch

from data import BenchmarkData, CONDITIONS, ROOT, SEED, resize_for_model
from run_benchmark import Experiment, atomic_write, target_data, training_data, write_csv
from training import ARCHITECTURES, POLICIES, Trainer

PRIMARY = ROOT / "deeplearning" / "results" / "benchmark"
SECONDARY = ROOT / "Dataset_Sekunder"
DEFAULT_OUTPUT = ROOT / "deeplearning" / "results" / "external_evaluation"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_recipes():
    source_config = json.loads((PRIMARY / "run_config.json").read_text())
    for name, expected in source_config["sources"].items():
        source = ROOT / "deeplearning" / ("splits" if name.endswith(".csv") else "") / name
        if digest(source) != expected:
            raise ValueError(f"Primary benchmark source changed: {name}")

    tuning = defaultdict(lambda: defaultdict(list))
    seen_folds = set()
    for line in (PRIMARY / "folds.jsonl").read_text().splitlines():
        row = json.loads(line)
        config = Experiment(**row["config"])
        key = (config, row["fold"])
        if key in seen_folds:
            raise ValueError(f"Duplicate primary outer fold: {key}")
        seen_folds.add(key)
        for fit in row["inner"]:
            tuning[config][fit["learning_rate"]].append(fit)

    recipes = []
    for values in product(CONDITIONS, ARCHITECTURES, POLICIES, (False, True)):
        config = Experiment(*values)
        candidates = tuning[config]
        if set(candidates) != set(POLICIES[config.initialization].learning_rates):
            raise ValueError(f"Unexpected primary tuning candidates for {config}")
        if any(len(fits) != 15 for fits in candidates.values()):
            raise ValueError(f"Incomplete primary inner-fold results for {config}")
        lr, selected = max(candidates.items(), key=lambda item: (
            statistics.mean(fit["macro_f1"] for fit in item[1]),
            -statistics.mean(fit["best_loss"] for fit in item[1]),
        ))
        epochs = int(statistics.median(fit["best_epoch"] for fit in selected))
        recipes.append(dict(**vars(config), learning_rate=lr, epochs=epochs))
    if len(seen_folds) != 180:
        raise ValueError("Expected all 180 primary outer-fold results")
    return recipes


def secondary_inputs(classes, device):
    with (SECONDARY / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if Counter(row["label"] for row in rows) != {label: 25 for label in classes}:
        raise ValueError("Expected 25 secondary photos per class")
    for field in ("image", "group_id", "sha256"):
        if not all(row[field] for row in rows) or len({row[field] for row in rows}) != 150:
            raise ValueError(f"Secondary metadata must contain 150 unique {field} values")

    images = []
    labels = []
    for row in rows:
        if row["split"] != "evaluation" or Path(row["image"]).name != row["image"]:
            raise ValueError("Unexpected secondary file or split")
        path = SECONDARY / row["label"] / row["image"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Secondary photo changed: {row['image']}")
        with Image.open(path) as source:
            # No condition-dependent processing or manual rotation on secondary photos.
            images.append(resize_for_model(np.asarray(source.convert("RGB"))))
        labels.append(classes.index(row["label"]))
    batch = torch.from_numpy(np.stack(images)).to(device)
    return rows, batch, torch.tensor(labels, device=device)


def reports(output, results, classes):
    metrics = []
    predictions = []
    per_class = []
    paired = []
    seen = {}
    for result in results:
        config = result["config"]
        seen[Experiment(**{key: config[key] for key in Experiment.__dataclass_fields__})] = result
        metrics.append(dict(**config, **result["metrics"]))
        for prediction in result["predictions"]:
            scores = {f"p_{label}": value for label, value in zip(classes, prediction["probabilities"])}
            predictions.append(dict(
                **config, image=prediction["image"], observation_id=prediction["observation_id"],
                true=prediction["true"], pred=prediction["pred"], correct=prediction["correct"], **scores,
            ))
        for label in classes:
            subset = [row for row in result["predictions"] if row["true"] == label]
            correct = sum(row["correct"] for row in subset)
            per_class.append(dict(**config, label=label, correct=correct, photos=len(subset),
                                  recall=correct / len(subset)))

    for config, result in seen.items():
        if config.input == "N":
            continue
        baseline = seen.get(Experiment("N", config.architecture, config.initialization, config.augmentation))
        if baseline is None:
            continue
        before = {row["image"]: row["correct"] for row in baseline["predictions"]}
        after = {row["image"]: row["correct"] for row in result["predictions"]}
        if before.keys() != after.keys():
            raise ValueError("Paired predictions disagree on test membership")
        paired.append(dict(
            **vars(config), reference="N",
            macro_f1_delta=result["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
            accuracy_delta=result["metrics"]["accuracy"] - baseline["metrics"]["accuracy"],
            only_pcd_correct=sum(after[image] and not before[image] for image in before),
            only_n_correct=sum(before[image] and not after[image] for image in before),
        ))
    metrics.sort(key=lambda item: -item["macro_f1"])
    write_csv(output / "summary.csv", metrics)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "recall_per_class.csv", per_class)
    write_csv(output / "pcd_deltas.csv", paired)


def run(output, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    data = BenchmarkData()
    if len(data.ids) != 60 or len(data.classes) != 6:
        raise ValueError("Expected 60 primary training images and six classes")
    recipes = locked_recipes()
    output.mkdir(parents=True, exist_ok=True)
    labels = data.labels.to(device)
    train_ids = list(range(len(data.ids)))
    trainer = Trainer(device, len(data.classes))

    if smoke:
        # Use only primary inputs to test the training path before external evaluation.
        temporary = data.input_bank("E", device)
        pictures, targets = training_data(temporary, labels, train_ids, False)
        _, fit = trainer.fit("resnet50", "pretrained", 1e-4, 1, SEED, pictures, targets)
        print(f"Primary-only smoke passed on {torch.cuda.get_device_name(device)}, "
              f"training loss {fit['curve'][0]['train_loss']:.4f}", flush=True)
        return

    recipe_info = dict(
        seed=SEED, train_photos=60, evaluation_photos=150, train_source="Dataset",
        evaluation_source="Dataset_Sekunder", pcd_scope="train_only",
        test_transform="RGB decode, shared 224 resize/pad and ImageNet tensor normalization",
        primary_tuning_sha256=digest(PRIMARY / "folds.jsonl"),
        secondary_manifest_sha256=digest(SECONDARY / "metadata.csv"),
        runner_sha256=digest(Path(__file__)), torch=version("torch"), gpu=torch.cuda.get_device_name(device),
        recipes=recipes,
    )
    config_path = output / "run_config.json"
    recipe_text = json.dumps(recipe_info, indent=2) + "\n"
    with (output / "runs.jsonl").open("a+b") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal.seek(0)
        previous = journal.read()
        if config_path.exists() and config_path.read_text() != recipe_text:
            raise ValueError("The frozen run configuration changed")
        if previous and not config_path.exists():
            raise ValueError("Existing results have no verified run configuration")
        atomic_write(config_path, lambda stream: stream.write(recipe_text))

        # Lock and save every training recipe before opening the external test images.
        rows, test_batch, test_y = secondary_inputs(data.classes, device)
        end = previous.rfind(b"\n") + 1
        if end != len(previous):
            journal.truncate(end)
        results = [json.loads(line) for line in previous[:end].splitlines()]
        completed = {tuple(record["config"].values()) for record in results}
        reports(output, results, data.classes)
        print(f"External benchmark start: {len(results)}/36 configurations complete", flush=True)
        for condition in CONDITIONS:
            own_recipes = [config for config in recipes
                           if config["input"] == condition and tuple(config.values()) not in completed]
            if not own_recipes:
                continue
            bank = data.input_bank(condition, device)
            for config in own_recipes:
                tag = (config["input"], config["architecture"], config["initialization"], config["augmentation"])
                print(f"Training {tag}: lr={config['learning_rate']}, epochs={config['epochs']}", flush=True)
                pictures, targets = training_data(bank, labels, train_ids, config["augmentation"])
                model, fit = trainer.fit(
                    config["architecture"], config["initialization"], config["learning_rate"],
                    config["epochs"], SEED, pictures, targets,
                )
                loss, f1, accuracy, pred, probabilities = trainer.evaluate(model, test_batch, test_y)
                del model, pictures, targets
                predictions = [dict(
                    image=row["image"], observation_id=row["observation_id"], true=row["label"],
                    pred=data.classes[index], correct=row["label"] == data.classes[index],
                    probabilities=probs,
                ) for row, index, probs in zip(rows, pred, probabilities)]
                result = dict(config=config, metrics=dict(
                    photos=len(predictions), correct=sum(row["correct"] for row in predictions),
                    macro_f1=f1, accuracy=accuracy, loss=loss, train_seconds=fit["seconds"],
                ), training_curve=fit["curve"], predictions=predictions)
                journal.write(json.dumps(result, separators=(",", ":")).encode() + b"\n")
                journal.flush()
                os.fsync(journal.fileno())
                results.append(result)
                reports(output, results, data.classes)
                print(f"External {len(results)}/36 {tag}: correct={result['metrics']['correct']}/150 "
                      f"macro-f1={f1:.4f}", flush=True)
            del bank
    print("External benchmark complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Test one epoch using primary images only")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output, args.smoke)


if __name__ == "__main__":
    main()
