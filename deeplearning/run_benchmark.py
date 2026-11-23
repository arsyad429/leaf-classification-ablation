"""Run the stratified nested-CV deep learning benchmark on the ML folds."""

import argparse
from dataclasses import asdict, dataclass
import csv
import fcntl
import hashlib
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import statistics
from tempfile import NamedTemporaryFile

import torch

from data import BenchmarkData, CONDITIONS, ROOT, SEED, SIZE, SPLITS
from training import ARCHITECTURES, BATCH_SIZE, POLICIES, WEIGHT_ENUMS, Trainer, scores


@dataclass(frozen=True)
class Experiment:
    input: str
    architecture: str
    initialization: str
    augmentation: bool


def atomic_write(path, write):
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as staged:
        temporary = Path(staged.name)
        try:
            write(staged)
            staged.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_csv(path, rows):
    if not rows:
        return

    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, write)


def target_data(bank, labels, indices):
    return bank.originals[indices], labels[indices]


def training_data(bank, labels, indices, augmentation):
    images = bank.training_variants(indices) if augmentation else bank.originals[indices].unsqueeze(1)
    return images, labels[indices]


def run_fold(config, fold, data, bank, labels, trainer):
    policy = POLICIES[config.initialization]
    tuning = []
    tag = f"{config.input}/{config.architecture}/{config.initialization}/aug={config.augmentation}"
    for lr in policy.learning_rates:
        for inner_fold in range(3):
            train_indices = data.inner[fold, inner_fold, "train"]
            val_indices = data.inner[fold, inner_fold, "validation"]
            train, train_y = training_data(bank, labels, train_indices, config.augmentation)
            validation = target_data(bank, labels, val_indices)
            seed = SEED + 100 * (fold + 1) + inner_fold
            model, metric = trainer.fit(
                config.architecture, config.initialization, lr, policy.max_epochs, seed,
                train, train_y, validation=validation, patience=policy.patience,
            )
            del model, train, train_y, validation
            tuning.append(dict(learning_rate=lr, inner_fold=inner_fold, **metric))
            print(f"{tag} outer={fold} inner={inner_fold} lr={lr:g} epochs={metric['epochs']} "
                  f"best={metric['best_epoch']} f1={metric['macro_f1']:.4f}", flush=True)
    groups = [[record for record in tuning if record["learning_rate"] == lr] for lr in policy.learning_rates]
    selected = max(groups, key=lambda group: (
        statistics.mean(record["macro_f1"] for record in group),
        -statistics.mean(record["best_loss"] for record in group),
    ))
    lr = selected[0]["learning_rate"]
    epochs = int(statistics.median(record["best_epoch"] for record in selected))
    train, train_y = training_data(bank, labels, data.outer[fold, "train"], config.augmentation)
    evaluation = data.outer[fold, "evaluation"]
    model, final = trainer.fit(
        config.architecture, config.initialization, lr, epochs, SEED + 100 * (fold + 1) + 99,
        train, train_y,
    )
    loss, f1, accuracy, predictions, probabilities = trainer.evaluate(
        model, *target_data(bank, labels, evaluation),
    )
    del model
    predictions = [
        dict(image=data.ids[image_index], true=data.classes[data.labels[image_index]],
             pred=data.classes[prediction], correct=int(data.labels[image_index]) == prediction,
             probabilities=probability)
        for image_index, prediction, probability in zip(evaluation, predictions, probabilities)
    ]
    return dict(config=asdict(config), fold=fold, selected_lr=lr, refit_epochs=epochs,
                loss=loss, macro_f1=f1, accuracy=accuracy, predictions=predictions,
                inner=tuning, refit=final)


def export_results(output, results, classes):
    prediction_rows = []
    summaries = []
    names = {name: index for index, name in enumerate(classes)}
    for config, folds in results.items():
        config_fields = asdict(config)
        predictions = []
        true = []
        for fold, row in sorted(folds.items()):
            for prediction in row["predictions"]:
                probability_columns = {
                    f"p_{label}": value for label, value in zip(classes, prediction["probabilities"])
                }
                prediction_rows.append(dict(
                    **config_fields, fold=fold, image=prediction["image"], true=prediction["true"],
                    pred=prediction["pred"], correct=prediction["correct"], **probability_columns,
                ))
                true.append(names[prediction["true"]])
                predictions.append(names[prediction["pred"]])
        if len(folds) == 5:
            metrics = list(folds.values())
            oof_f1, accuracy = scores(true, predictions, len(classes))
            summaries.append(dict(
                **config_fields, folds=len(folds), photos=len(true),
                macro_f1_mean=statistics.mean(row["macro_f1"] for row in metrics),
                macro_f1_std=statistics.stdev(row["macro_f1"] for row in metrics),
                oof_macro_f1=oof_f1, oof_accuracy=accuracy,
                correct=sum(y == prediction for y, prediction in zip(true, predictions)),
                refit_seconds=sum(row["refit"]["seconds"] for row in metrics),
                inner_epoch_limits=sum(run["stop_reason"] == "epoch_limit"
                                       for row in metrics for run in row["inner"]),
            ))
    summaries.sort(key=lambda row: (-row["macro_f1_mean"], row["architecture"], row["initialization"]))
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "oof_predictions.csv", prediction_rows)


def run_smoke(data, labels, trainer):
    for condition, architecture in zip(CONDITIONS, ARCHITECTURES):
        bank = data.input_bank(condition, labels.device)
        train_ids = data.outer[0, "train"][:8]
        validation = target_data(bank, labels, data.outer[0, "evaluation"][:8])
        for initialization, policy in POLICIES.items():
            train, train_y = training_data(bank, labels, train_ids, initialization == "pretrained")
            model, metric = trainer.fit(
                architecture, initialization, policy.learning_rates[0], 1, SEED,
                train, train_y, validation=validation, patience=policy.patience,
            )
            del model, train, train_y
            print(f"Smoke OK {condition} {architecture} {initialization}: "
                  f"validation loss={metric['best_loss']:.4f}", flush=True)
        del bank


def run_benchmark(output, data, labels, trainer):
    output.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("training.py"),
               *(SPLITS / name for name in ("outer.csv", "inner.csv", "images.csv")))
    manifest = dict(
        architectures=ARCHITECTURES, conditions=CONDITIONS,
        policies={name: asdict(policy) for name, policy in POLICIES.items()},
        pretrained={name: weight.url for name, weight in WEIGHT_ENUMS.items()},
        seed=SEED, batch_size=BATCH_SIZE, input_size=SIZE, classes=data.classes,
        precision="bfloat16 autocast" if trainer.use_amp else "float32",
        gpu=torch.cuda.get_device_name(labels.device),
        versions={"torch": version("torch"), "torchvision": version("torchvision"),
                  "sklearn": version("scikit-learn"), "opencv": __import__("cv2").__version__},
        sources={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    )
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    manifest_path = output / "run_config.json"
    with (output / "folds.jsonl").open("a+b") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal.seek(0)
        previous = journal.read()
        if previous and (not manifest_path.exists() or manifest_path.read_text() != manifest_text):
            raise RuntimeError("Run configuration changed. Use a new results directory.")
        atomic_write(manifest_path, lambda stream: stream.write(manifest_text))

        end = previous.rfind(b"\n") + 1
        if end != len(previous):
            journal.truncate(end)
        results = {}
        for line in previous[:end].splitlines():
            row = json.loads(line)
            config = Experiment(**row["config"])
            results.setdefault(config, {})[row["fold"]] = row
        export_results(output, results, data.classes)
        completed = sum(map(len, results.values()))
        total = len(CONDITIONS) * len(ARCHITECTURES) * len(POLICIES) * 2 * 5
        print(f"Benchmark start: {completed}/{total} outer fits already complete", flush=True)
        for condition in CONDITIONS:
            configs = [Experiment(condition, model, initialization, augmentation) for model, initialization, augmentation
                       in product(ARCHITECTURES, POLICIES, (False, True))]
            if all(len(results.get(config, {})) == 5 for config in configs):
                continue
            bank = data.input_bank(condition, labels.device)
            for config in configs:
                folds = results.setdefault(config, {})
                for fold in range(5):
                    if fold in folds:
                        continue
                    row = run_fold(config, fold, data, bank, labels, trainer)
                    journal.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    folds[fold] = row
                    export_results(output, results, data.classes)
                    completed += 1
                    print(f"Outer {completed}/{total} {config} fold={fold} "
                          f"f1={row['macro_f1']:.4f} accuracy={row['accuracy']:.4f}", flush=True)
            del bank
    print("Benchmark complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Check all models, weight modes and image treatments")
    parser.add_argument("--output", type=Path, default=ROOT / "deeplearning" / "results" / "benchmark")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    data = BenchmarkData()
    labels = data.labels.to(device)
    trainer = Trainer(device, len(data.classes))
    print(f"Device {torch.cuda.get_device_name(device)}; photos={len(labels)}; classes={len(data.classes)}", flush=True)
    if args.smoke:
        run_smoke(data, labels, trainer)
    else:
        run_benchmark(args.output, data, labels, trainer)


if __name__ == "__main__":
    main()
