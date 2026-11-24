"""Run Otsu extensions of the primary and external leaf classification benchmarks."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict
import fcntl
from itertools import product
import json
import os
from pathlib import Path
import statistics

import cv2
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from data import AUGMENTATIONS, ROOT, SEED, SIZE, BenchmarkData, augment, preprocess, resize_for_model
from run_benchmark import Experiment, atomic_write, export_results, run_fold, training_data, write_csv
from run_external import SECONDARY, digest, locked_recipes, reports, secondary_inputs
from training import ARCHITECTURES, POLICIES, Trainer

CONDITIONS = ("S-O", "E-S-O")
DESTINATION = ROOT / "deeplearning/results/otsu"
HISTORY = ROOT / "deeplearning/results/external_evaluation"
PRIMARY_HISTORY = ROOT / "deeplearning/results/benchmark"
COMPARISONS = {"S-O": ("N", "S-K"), "E-S-O": ("E",)}


def otsu_rgb(image, condition):
    rgb = preprocess(image, "E") if condition == "E-S-O" else image
    channel = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1]
    threshold, mask = cv2.threshold(channel, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)
    return np.where(mask[:, :, None], rgb, 0), mask, threshold


class OtsuBank:
    def __init__(self, condition, photos, device):
        self.photos = photos
        self.condition = condition
        self.variants = {}
        self.originals = torch.from_numpy(np.stack([self.prepare(photo) for photo in photos])).to(device)

    def prepare(self, photo):
        return resize_for_model(otsu_rgb(photo, self.condition)[0])

    def training_variants(self, indices):
        for index in indices:
            if index in self.variants:
                continue
            versions = [self.originals[index]]
            for version, operation in enumerate(AUGMENTATIONS, 1):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, index, version, 0]))
                photo = augment(self.photos[index], operation, rng)
                versions.append(torch.from_numpy(np.array(self.prepare(photo), copy=True)).to(self.originals.device))
            self.variants[index] = torch.stack(versions)
        return torch.stack([self.variants[index] for index in indices])


def experiments():
    return [Experiment(*values) for values in product(CONDITIONS, ARCHITECTURES, POLICIES, (False, True))]


def manifest(mode, device, extra):
    sources = ROOT / "deeplearning"
    paths = [Path(__file__), *(sources / name for name in
             ("data.py", "training.py", "run_benchmark.py", "run_external.py",
              "splits/images.csv", "splits/outer.csv", "splits/inner.csv"))]
    return dict(stage=mode, conditions=CONDITIONS, seed=SEED, size=SIZE,
                architectures=ARCHITECTURES, gpu=torch.cuda.get_device_name(device),
                torch=torch.__version__, opencv=cv2.__version__,
                sources={path.name: digest(path) for path in paths}, **extra)


@contextmanager
def open_journal(output, name, configuration):
    output.mkdir(parents=True, exist_ok=True)
    with (output / name).open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        previous = stream.read()
        manifest_path = output / "run_config.json"
        manifest_text = json.dumps(configuration, indent=2) + "\n"
        if manifest_path.exists() and manifest_path.read_text() != manifest_text:
            raise ValueError("Existing Otsu run has different settings or sources")
        atomic_write(manifest_path, lambda handle: handle.write(manifest_text))
        last_complete = previous.rfind(b"\n") + 1
        if last_complete != len(previous):
            stream.truncate(last_complete)
        rows = [json.loads(line) for line in previous[:last_complete].splitlines()]
        yield stream, rows


def persist(stream, row):
    stream.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_primary(data, labels, trainer, output):
    sources = manifest("primary_oof", labels.device, {"outer_folds": 5, "inner_folds": 3,
                        "pcd_scope": "train_validation_evaluation",
                        "policies": {name: asdict(policy) for name, policy in POLICIES.items()}})
    with open_journal(output, "folds.jsonl", sources) as (stream, rows):
        results = {}
        for row in rows:
            results.setdefault(Experiment(**row["config"]), {})[row["fold"]] = row
        export_results(output, results, data.classes)
        completed = len(rows)
        print(f"Otsu primary start: {completed}/120 outer fits", flush=True)
        for condition in CONDITIONS:
            configs = [config for config in experiments() if config.input == condition]
            if all(len(results.get(config, {})) == 5 for config in configs):
                continue
            bank = OtsuBank(condition, data.photos, labels.device)
            for config in configs:
                folds = results.setdefault(config, {})
                for fold in range(5):
                    if fold in folds:
                        continue
                    row = run_fold(config, fold, data, bank, labels, trainer)
                    persist(stream, row)
                    folds[fold] = row
                    export_results(output, results, data.classes)
                    completed += 1
                    print(f"Otsu outer {completed}/120 {config} fold={fold} macro-f1={row['macro_f1']:.4f}", flush=True)
            del bank
    print("Otsu primary complete", flush=True)


def recipes_from_primary(path):
    candidates = defaultdict(lambda: defaultdict(list))
    seen = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        config = Experiment(**row["config"])
        fold = (config, row["fold"])
        if fold in seen:
            raise ValueError("Duplicate outer fold in primary journal")
        seen.add(fold)
        for trial in row["inner"]:
            candidates[config][trial["learning_rate"]].append(trial)
    if len(seen) != 120:
        raise ValueError("Otsu primary evaluation must finish before training for external evaluation")
    recipes = []
    for config in experiments():
        levels = candidates[config]
        if set(levels) != set(POLICIES[config.initialization].learning_rates) or any(len(x)!=15 for x in levels.values()):
            raise ValueError(f"Incomplete Otsu tuning: {config}")
        lr, fits = max(levels.items(), key=lambda item: (
            statistics.mean(fit["macro_f1"] for fit in item[1]),
            -statistics.mean(fit["best_loss"] for fit in item[1]),
        ))
        epochs = int(statistics.median(fit["best_epoch"] for fit in fits))
        recipes.append(dict(**asdict(config), learning_rate=lr, epochs=epochs))
    return recipes


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def history_data():
    signature = json.loads((HISTORY / "run_config.json").read_text())
    if signature["secondary_manifest_sha256"] != digest(SECONDARY / "metadata.csv"):
        raise ValueError("Existing external benchmark used different secondary photos")
    fields = ("input", "architecture", "initialization", "augmentation")
    scores = {tuple(r[k] for k in fields): float(r["macro_f1"]) for r in read_rows(HISTORY / "summary.csv")}
    history = defaultdict(dict)
    for row in read_rows(HISTORY / "predictions.csv"):
        history[tuple(row[k] for k in fields)][row["image"]] = (row["true"], row["pred"])
    if len(scores) != 36 or any(len(x) != 150 for x in history.values()):
        raise ValueError("Historical external results are incomplete")
    return scores, history


def external_comparisons(output, results, baseline_scores, baseline_predictions):
    paired = []
    for row in results:
        config = row["config"]
        current = {p["image"]: (p["true"],p["pred"]) for p in row["predictions"]}
        for reference in COMPARISONS[config["input"]]:
            key = (reference, config["architecture"], config["initialization"], str(config["augmentation"]))
            old = baseline_predictions[key]
            if old.keys() != current.keys() or any(old[i][0]!=current[i][0] for i in current):
                raise ValueError(f"Historical evaluation membership differs for {key}")
            before = {i: labels[0]==labels[1] for i,labels in old.items()}
            after = {i: labels[0]==labels[1] for i,labels in current.items()}
            paired.append(dict(
                **config, reference=reference,
                macro_f1_delta=row["metrics"]["macro_f1"]-baseline_scores[key],
                correct_delta=sum(after.values())-sum(before.values()),
                only_otsu_correct=sum(after[i] and not before[i] for i in current),
                only_reference_correct=sum(before[i] and not after[i] for i in current),
            ))
    write_csv(output / "pcd_deltas.csv", paired)


def run_external(data, labels, trainer, output, primary_journal):
    recipes = recipes_from_primary(primary_journal)
    scores, history = history_data()
    configuration = manifest("external", labels.device, dict(
        train_photos=60, evaluation_photos=150, pcd_scope="train_only",
        primary_journal=digest(primary_journal), secondary_manifest_sha256=digest(SECONDARY/"metadata.csv"),
        test_transform="RGB decode, shared 224 resize/pad, and ImageNet tensor normalization",
        recipes=recipes,
    ))
    with open_journal(output, "runs.jsonl", configuration) as (stream, results):
        rows, test_images, test_y = secondary_inputs(data.classes, labels.device)
        completed = {tuple(row["config"].values()) for row in results}
        reports(output, results, data.classes)
        external_comparisons(output, results, scores, history)
        print(f"Otsu external start: {len(results)}/24 configurations", flush=True)
        photo_ids = list(range(len(data.ids)))
        for condition in CONDITIONS:
            selected = [recipe for recipe in recipes if recipe["input"] == condition and tuple(recipe.values()) not in completed]
            if not selected:
                continue
            bank = OtsuBank(condition, data.photos, labels.device)
            for config in selected:
                pictures, train_y = training_data(bank, labels, photo_ids, config["augmentation"])
                print(f"Otsu external train {config}", flush=True)
                model, fit = trainer.fit(config["architecture"], config["initialization"], config["learning_rate"],
                                         config["epochs"], SEED, pictures, train_y)
                loss, f1, accuracy, pred, probabilities = trainer.evaluate(model, test_images, test_y)
                del model, pictures, train_y
                predictions = [
                    dict(image=row["image"], observation_id=row["observation_id"], true=row["label"],
                         pred=data.classes[index], correct=row["label"] == data.classes[index], probabilities=probs)
                    for row,index,probs in zip(rows,pred,probabilities)
                ]
                result = dict(config=config, metrics=dict(
                    photos=len(predictions), correct=sum(row["correct"] for row in predictions),
                    macro_f1=f1, accuracy=accuracy, loss=loss, train_seconds=fit["seconds"],
                ), training_curve=fit["curve"], predictions=predictions)
                persist(stream,result)
                results.append(result)
                reports(output,results,data.classes)
                external_comparisons(output,results,scores,history)
                print(f"Otsu external {len(results)}/24 macro-f1={f1:.4f}", flush=True)
            del bank
    print("Otsu external complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "benchmark", "external_evaluation"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    cv2.setNumThreads(1)
    device = torch.device("cuda:0")
    data = BenchmarkData()
    labels = data.labels.to(device)
    trainer = Trainer(device, len(data.classes))
    if len(data.ids) != 60:
        raise ValueError("Expected 60 primary photos")
    locked_recipes()  # Verify the original run's shared code and split hashes.
    with threadpool_limits(limits=2):
        if args.stage == "smoke":
            for condition in CONDITIONS:
                bank = OtsuBank(condition, data.photos, device)
                val_ids = data.inner[0,0,"validation"][:8]
                train_ids = data.inner[0,0,"train"][:8]
                x,y = training_data(bank, labels, train_ids, condition=="E-S-O")
                model, fit = trainer.fit("resnet50","pretrained",1e-4,1,SEED,x,y,
                                         (bank.originals[val_ids],labels[val_ids]),patience=1)
                del model
                print(f"Otsu smoke passed {condition}, validation loss {fit['best_loss']:.4f}",flush=True)
        elif args.stage == "benchmark":
            run_primary(data, labels, trainer, DESTINATION / args.stage)
        else:
            run_external(data, labels, trainer, DESTINATION / args.stage, DESTINATION / "benchmark/folds.jsonl")


if __name__ == "__main__":
    main()
