"""Model construction, training, early stopping and scoring."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch import nn
from torchvision import models


ARCHITECTURES = ("resnet50", "vit_b_16", "swin_t")
WEIGHT_ENUMS = {
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
    "swin_t": models.Swin_T_Weights.IMAGENET1K_V1,
}
BATCH_SIZE = 8


@dataclass(frozen=True)
class Policy:
    learning_rates: tuple
    max_epochs: int
    patience: int


POLICIES = {
    "pretrained": Policy((3e-5, 1e-4), 20, 4),
    "scratch": Policy((3e-4, 1e-3), 40, 6),
}


def build_model(architecture, initialization, classes):
    weights = WEIGHT_ENUMS[architecture] if initialization == "pretrained" else None
    model = getattr(models, architecture)(weights=weights)
    if architecture == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif architecture == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, classes)
    else:
        model.head = nn.Linear(model.head.in_features, classes)
    return model


def scores(labels, predictions, classes):
    bins = np.asarray(labels) * classes + np.asarray(predictions)
    matrix = np.bincount(bins, minlength=classes ** 2).reshape(classes, classes)
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * matrix.diagonal(), denom, out=np.zeros(classes), where=denom > 0)
    return float(f1.mean()), float(matrix.trace() / matrix.sum())


class Trainer:
    def __init__(self, device, class_count):
        self.device = device
        self.class_count = class_count
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.loss = nn.CrossEntropyLoss()
        self.use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def precision(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_amp else nullcontext()

    def inputs(self, batch):
        return (batch.float().div_(255) - self.mean) / self.std

    @torch.inference_mode()
    def evaluate(self, model, pictures, labels):
        model.eval()
        probabilities = []
        total_loss = torch.zeros((), device=self.device)
        for batch, target in zip(pictures.split(BATCH_SIZE), labels.split(BATCH_SIZE)):
            with self.precision():
                logits = model(self.inputs(batch))
                total_loss += self.loss(logits, target) * len(target)
            probabilities.append(logits.float().softmax(dim=1))
        probabilities = torch.cat(probabilities).cpu()
        predictions = probabilities.argmax(dim=1).tolist()
        loss = float(total_loss.item() / len(labels))
        if not math.isfinite(loss):
            raise FloatingPointError("Validation or evaluation loss is not finite")
        f1, accuracy = scores(labels.cpu().tolist(), predictions, self.class_count)
        return loss, f1, accuracy, predictions, probabilities.tolist()

    def fit(self, architecture, initialization, lr, epochs, seed,
            train, train_y, validation=None, patience=None):
        torch.manual_seed(seed)
        model = build_model(architecture, initialization, self.class_count).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        rng = np.random.default_rng(seed)
        variant_counts = train.shape[1]
        default_indices = np.arange(len(train))
        offsets = np.random.default_rng(seed + 1).integers(variant_counts, size=len(train))
        curve = []
        best_loss = math.inf
        best_epoch = None
        best_f1 = None
        stop_reason = "epoch_limit"
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(default_indices)
            variants = (offsets + epoch - 1) % variant_counts
            total_loss = torch.zeros((), device=self.device)
            for start in range(0, len(order), BATCH_SIZE):
                picked = order[start:start + BATCH_SIZE]
                batch = train[picked, variants[picked]]
                target = train_y[picked]
                optimizer.zero_grad(set_to_none=True)
                with self.precision():
                    train_loss = self.loss(model(self.inputs(batch)), target)
                train_loss.backward()
                optimizer.step()
                total_loss += train_loss.detach() * len(target)
            loss = float(total_loss.item() / len(train))
            if not math.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            point = {"epoch": epoch, "train_loss": loss}
            if validation is not None:
                val_loss, val_f1, _, _, _ = self.evaluate(model, *validation)
                point.update(val_loss=val_loss, val_macro_f1=val_f1)
                if val_loss < best_loss:
                    best_loss, best_epoch, best_f1 = val_loss, epoch, val_f1
            curve.append(point)
            if patience is not None and epoch - best_epoch >= patience:
                stop_reason = "early_stopping"
                break
        return model, dict(
            best_epoch=best_epoch, best_loss=best_loss if best_epoch is not None else None,
            macro_f1=best_f1, stop_reason=stop_reason, epochs=len(curve),
            seconds=time.perf_counter() - started, curve=curve,
        )
