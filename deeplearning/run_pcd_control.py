"""Run a matched train-only PCD comparison on the fixed secondary test set."""

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import statistics

import torch

from data import BenchmarkData, CONDITIONS, ROOT
from run_benchmark import Experiment, atomic_write, training_data, write_csv
from run_external import SECONDARY, digest, locked_recipes, secondary_inputs
from training import Trainer

OUTPUT = ROOT / "deeplearning/results/pcd_control"
BASE = Experiment("N", "swin_t", "pretrained", False)
SEEDS = (42, 43, 44)


def export(output, records):
    write_csv(output / "summary.csv", [
        dict(input=row["input"], seed=row["seed"], **row["metrics"]) for row in records
    ])
    write_csv(output / "predictions.csv", [
        dict(input=row["input"], seed=row["seed"], **prediction)
        for row in records for prediction in row["predictions"]
    ])
    indexed = {(r["input"], r["seed"]): r for r in records}
    pairs = []
    for condition in CONDITIONS[1:]:
        for seed in SEEDS:
            original = indexed.get(("N", seed))
            changed = indexed.get((condition, seed))
            if original is None or changed is None:
                continue
            before = {row["image"]: row["correct"] for row in original["predictions"]}
            after = {row["image"]: row["correct"] for row in changed["predictions"]}
            if before.keys() != after.keys():
                raise ValueError("Paired runs evaluated different photos")
            pairs.append(dict(
                input=condition, seed=seed,
                macro_f1_delta=changed["metrics"]["macro_f1"] - original["metrics"]["macro_f1"],
                correct_delta=changed["metrics"]["correct"] - original["metrics"]["correct"],
                only_pcd_correct=sum(after[k] and not before[k] for k in before),
                only_n_correct=sum(before[k] and not after[k] for k in before),
            ))
    write_csv(output / "deltas_per_seed.csv", pairs)
    aggregate = []
    for condition in CONDITIONS[1:]:
        subset = [row for row in pairs if row["input"] == condition]
        if len(subset) == len(SEEDS):
            delta = [row["macro_f1_delta"] for row in subset]
            aggregate.append(dict(input=condition, seeds=len(delta), mean_delta=statistics.mean(delta),
                                  sd_delta=statistics.stdev(delta), seeds_improved=sum(x > 0 for x in delta)))
    write_csv(output / "deltas_summary.csv", aggregate)


def run(output):
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    primary = BenchmarkData()
    if len(primary.ids) != 60:
        raise ValueError("Expected all 60 primary photos")
    baseline = next(r for r in locked_recipes() if all(r[k] == v for k,v in asdict(BASE).items()))
    lr, epochs = baseline["learning_rate"], baseline["epochs"]
    if (lr, epochs) != (1e-4, 12):
        raise ValueError("The primary baseline no longer matches the locked control recipe")
    trainer = Trainer(device, len(primary.classes))
    labels = primary.labels.to(device)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        architecture=BASE.architecture, initialization=BASE.initialization, augmentation=BASE.augmentation,
        conditions=CONDITIONS, seeds=SEEDS, learning_rate=lr, epochs=epochs, train_photos=60,
        test_photos=150, pcd_scope="train_only", secondary_manifest_sha256=digest(SECONDARY / "metadata.csv"),
        control_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(device),
    )
    config_text = json.dumps(config, indent=2) + "\n"
    with (output / "runs.jsonl").open("a+b") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal.seek(0)
        previous = journal.read()
        config_path = output / "config.json"
        if previous and (not config_path.exists() or config_path.read_text() != config_text):
            raise ValueError("Refusing to reuse results from a different control recipe")
        atomic_write(config_path, lambda stream: stream.write(config_text))
        end = previous.rfind(b"\n") + 1
        if end != len(previous):
            journal.truncate(end)
        records = [json.loads(line) for line in previous[:end].splitlines()]
        completed = {(r["input"],r["seed"]) for r in records}
        export(output, records)
        metadata, test_images, test_labels = secondary_inputs(primary.classes, device)
        print(f"PCD control start: {len(records)}/9 complete", flush=True)
        train_ids = list(range(len(primary.ids)))
        for condition in CONDITIONS:
            if all((condition,seed) in completed for seed in SEEDS):
                continue
            bank = primary.input_bank(condition, device)
            pictures, targets = training_data(bank, labels, train_ids, False)
            for seed in SEEDS:
                if (condition,seed) in completed:
                    continue
                print(f"Training {condition}, seed {seed}, lr={lr:g}, epochs={epochs}", flush=True)
                model, fit = trainer.fit(BASE.architecture, BASE.initialization, lr, epochs, seed, pictures, targets)
                loss, f1, accuracy, preds, _ = trainer.evaluate(model, test_images, test_labels)
                del model
                predictions = [dict(image=row["image"], true=row["label"], pred=primary.classes[index],
                                    correct=row["label"] == primary.classes[index])
                               for row,index in zip(metadata,preds)]
                result = dict(input=condition, seed=seed,
                              metrics=dict(photos=len(predictions), correct=sum(x["correct"] for x in predictions),
                                           macro_f1=f1, accuracy=accuracy, loss=loss, train_seconds=fit["seconds"]),
                              training_curve=fit["curve"], predictions=predictions)
                journal.write(json.dumps(result,separators=(",",":")).encode()+b"\n")
                journal.flush()
                os.fsync(journal.fileno())
                records.append(result)
                export(output, records)
                print(f"Control {len(records)}/9 {condition} seed={seed} "
                      f"correct={result['metrics']['correct']}/150 macro-f1={f1:.4f}", flush=True)
            del bank, pictures, targets
    print("PCD control complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    run(parser.parse_args().output)
