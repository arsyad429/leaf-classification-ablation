"""Refit stored DL experiments and export Grad-CAM on evaluation photos."""

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

from data import BenchmarkData, ROOT, SEED
from run_benchmark import training_data, write_csv
from run_external import secondary_inputs
from run_otsu import CONDITIONS as OTSU_CONDITIONS, OtsuBank
from training import Trainer, build_model

RESULTS = ROOT / "deeplearning" / "results"
OUTPUT = RESULTS / "gradcam"
SCOPES = {
    "benchmark": ("benchmark", "folds.jsonl"),
    "external_evaluation": ("external_evaluation", "runs.jsonl"),
    "otsu_benchmark": ("otsu/benchmark", "folds.jsonl"),
    "otsu_external_evaluation": ("otsu/external_evaluation", "runs.jsonl"),
}
CREDIT_FIELDS = ("image", "label", "source_url", "photo_url", "license_code", "license_url", "attribution")


def activation_layer(model, architecture):
    if architecture == "resnet50":
        return model.layer4[-1], "layer4[-1]"
    if architecture == "vit_b_16":
        return model.encoder.layers[-1].ln_1, "encoder.layers[-1].ln_1 (patch tokens)"
    if architecture == "swin_t":
        return model.features[-1][-1].norm1, "features[-1][-1].norm1"
    raise ValueError(f"Unsupported architecture: {architecture}")


def spatial(activation, architecture):
    if architecture == "vit_b_16":
        tokens = activation[:, 1:, :]
        side = int(tokens.shape[1] ** .5)
        if tokens.shape[1] != side * side:
            raise ValueError("ViT patch token count is not square")
        return tokens.reshape(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)
    if architecture == "swin_t":
        return activation.permute(0, 3, 1, 2)
    return activation


def cam(model, trainer, architecture, images, targets):
    model.eval()
    layer, layer_name = activation_layer(model, architecture)
    captured = []
    hook = layer.register_forward_hook(lambda _layer, _inputs, out: captured.append(out))
    try:
        with torch.enable_grad(), trainer.precision():
            logits = model(trainer.inputs(images))
            score = logits.gather(1, targets[:, None]).float().sum()
            gradients, = torch.autograd.grad(score, captured[0])
        features = spatial(captured[0].detach().float(), architecture)
        grads = spatial(gradients.detach().float(), architecture)
        maps = (features * grads.mean(dim=(2, 3), keepdim=True)).sum(dim=1, keepdim=True).relu()
        maps = F.interpolate(maps, size=tuple(images.shape[-2:]), mode="bilinear", align_corners=False)[:, 0]
        if not torch.isfinite(maps).all().item():
            raise ValueError(f"Non-finite CAM from {layer_name}")
        minimum = maps.amin(dim=(1, 2), keepdim=True)
        scale = maps.amax(dim=(1, 2), keepdim=True) - minimum
        normalized = (maps - minimum) / scale.clamp_min(1e-12)
        return normalized.cpu().numpy(), scale.flatten().gt(1e-12).cpu().tolist(), layer_name
    finally:
        hook.remove()


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def montage(path, config, label, items, images, maps, valid, layer, agreement, total):
    w, h = 183, 183
    pair_width = 2 * w + 30
    columns = 3
    lines = (len(items) + columns - 1) // columns
    height = 114 + lines * 272 + 45
    canvas = Image.new("RGB", (columns * pair_width + 54, height), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    ink = "#172a3d"
    muted = "#455b71"
    aug = "ON" if config["augmentation"] else "OFF"
    title = f"Grad-CAM | {config['architecture']} | {config['input']} | {config['initialization']} | Aug {aug}"
    draw.text((25, 24), title, font=font(24, True), fill=ink)
    draw.text((25, 61), f"{label} | Refit: {agreement}/{total} prediksi cocok dengan arsip | Layer: {layer}", font=font(15), fill=muted)
    draw.text((25, 83), "Kiri: input model. Kanan: Grad-CAM untuk kelas prediksi refit.", font=font(14), fill=muted)

    for i, (item, data, attention, useful) in enumerate(zip(items, images, maps, valid)):
        x = 25 + (i % columns) * pair_width
        y = 110 + (i // columns) * 272
        rgb = data.transpose(1, 2, 0)
        original = Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
        if useful:
            color = cv2.applyColorMap(np.rint(attention * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
            alpha = (.65 * attention)[..., None]
            mixed = np.rint(rgb * (1-alpha) + color * alpha).astype(np.uint8)
            overlay = Image.fromarray(mixed).resize((w, h), Image.Resampling.BILINEAR)
        else:
            overlay = original
        canvas.paste(original, (x, y))
        canvas.paste(overlay, (x+w+8, y))
        draw.text((x, y+h+7), item["image"], font=font(13, True), fill=ink)
        draw.text((x, y+h+27), f"Asli: {item['true']} | Refit: {item['refit_pred']}", font=font(12), fill=ink)
        note = f"Arsip: {item['archived_pred']} | CAM {'aktif' if useful else 'datar'}"
        draw.text((x, y+h+45), note, font=font(12), fill=muted)
    if total == 150:
        draw.text((25, height-35), "Foto iNaturalist ditampilkan dari input inferensi. Sumber dan atribusi: photo_credits.csv.", font=font(12), fill=muted)
    with NamedTemporaryFile(dir=path.parent, delete=False) as staging:
        temporary = Path(staging.name)
    try:
        canvas.save(temporary, format="JPEG", quality=88, optimize=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def use_rows(record, config, primary, secondary):
    archived = {row["image"]: row for row in record["predictions"]}
    if "fold" in record:
        fold = record["fold"]
        ids = primary.outer[fold, "evaluation"]
        names = [primary.ids[i] for i in ids]
        return archived, ids, names, SEED + 100*(fold+1) + 99, record["selected_lr"], record["refit_epochs"]
    names = [row["image"] for row in secondary]
    return archived, list(range(len(names))), names, SEED, config["learning_rate"], config["epochs"]


def display_cases(names, archived, predicted, reference):
    groups = defaultdict(list)
    for index, (name, refit) in enumerate(zip(names, predicted)):
        row = archived[name]
        if reference and reference[name]["license_code"] not in ("cc-by", "cc0"):
            continue
        groups[row["true"]].append((index, refit != row["true"]))
    chosen = [max(choices, key=lambda c:c[1])[0] for _,choices in sorted(groups.items())]
    if not chosen:
        raise ValueError("No attribution-compatible examples to render")
    return chosen


def load_scope(scope, output, inputs):
    folder, journal_name = SCOPES[scope]
    source = RESULTS / folder / journal_name
    records = [json.loads(line) for line in source.read_text().splitlines()]
    expected = 36 if not scope.startswith("otsu") else 24
    external = journal_name=="runs.jsonl"
    expected *= 1 if external else 5
    if len(records) != expected:
        raise ValueError(f"Incomplete source for {scope}: {len(records)} vs {expected}")
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "refits.csv"
    with (output / "manifest.json").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0)
        manifest = dict(scope=scope,source=folder+"/"+journal_name,
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        archived_model="refit from archived settings; match measured by predicted label agreement")
        cached = lock.read()
        if cached and json.loads(cached)!=manifest:
            raise ValueError("Grad-CAM outputs belong to another source or code version")
        if not cached:
            lock.write(json.dumps(manifest,indent=2)+"\n")
            lock.flush()
        run_records(scope,output,inputs,records,existing,external)


def run_records(scope, output, inputs, records, existing, external):
    primary, labels, trainer, secondary = inputs
    if external:
        s_rows, s_images, s_y = secondary
        reference = {row["image"]:row for row in s_rows}
    else:
        s_rows, reference = None, {}
    rows = []
    if existing.exists():
        with existing.open(encoding="utf-8",newline="") as stream:
            rows=list(csv.DictReader(stream))
    recorded={row["figure"] for row in rows}
    credit_rows = {}
    credit_file = output / "photo_credits.csv"
    if external and credit_file.exists():
        with credit_file.open(encoding="utf-8",newline="") as stream:
            credit_rows={row["image"]:row for row in csv.DictReader(stream)}
    present=defaultdict(list)
    for record in records: present[record["config"]["input"]].append(record)
    for condition, subset in present.items():
        bank = None
        for record in subset:
            cfg = record["config"]
            suffix = f"_fold{record['fold']+1}" if not external else ""
            name = f"{condition}_{cfg['architecture']}_{cfg['initialization']}_aug{int(cfg['augmentation'])}{suffix}.jpg"
            if name in recorded and (output / name).is_file():
                continue
            if bank is None:
                bank = (OtsuBank(condition,primary.photos,labels.device) if condition in OTSU_CONDITIONS
                        else primary.input_bank(condition,labels.device))
            archived, test_ids, names, seed, lr, epochs = use_rows(record,cfg,primary,s_rows)
            if set(archived)!=set(names):
                raise ValueError(f"Evaluation IDs differ from archived results: {name}")
            train_ids = list(range(len(primary.ids))) if external else primary.outer[record["fold"],"train"]
            x,y = training_data(bank,labels,train_ids,cfg["augmentation"])
            model,_ = trainer.fit(cfg["architecture"],cfg["initialization"],lr,epochs,seed,x,y)
            images, truth = (s_images,s_y) if external else (bank.originals[test_ids],labels[test_ids])
            _,macro_f1,accuracy,indexes,_ = trainer.evaluate(model,images,truth)
            predicted=[primary.classes[index] for index in indexes]
            agreement=sum(archived[n]["pred"]==v for n,v in zip(names,predicted))
            chosen=display_cases(names,archived,predicted,reference)
            maps, valid, layer=cam(model,trainer,cfg["architecture"],images[chosen],
                                  torch.tensor([indexes[i] for i in chosen],device=labels.device))
            items=[dict(image=names[i],true=archived[names[i]]["true"],
                        archived_pred=archived[names[i]]["pred"],refit_pred=predicted[i]) for i in chosen]
            label = "Evaluasi eksternal" if external else f"OOF primer, outer fold {record['fold']+1}"
            montage(output/name,cfg,label,items,images[chosen].cpu().numpy(),maps,valid,layer,agreement,len(names))
            if external:
                for item in items:
                    image=item["image"]
                    credit_rows[image]={k:reference[image][k] for k in CREDIT_FIELDS}
                write_csv(credit_file, list(credit_rows.values()))
            rows=[r for r in rows if r["figure"]!=name]
            rows.append(dict(
                figure=name,input=condition,architecture=cfg["architecture"],initialization=cfg["initialization"],
                augmentation=cfg["augmentation"],fold=record.get("fold",""),seed=seed,learning_rate=lr,epochs=epochs,
                matching_predictions=agreement,evaluation_photos=len(names),refit_macro_f1=macro_f1,
                refit_accuracy=accuracy,maps=len(valid),nonflat_maps=sum(valid),layer=layer,
            ))
            write_csv(existing,rows)
            print(f"Grad-CAM {scope} {len(rows)}/{len(records)} {name} agreement {agreement}/{len(names)} active {sum(valid)}/{len(valid)}",flush=True)
            del model,x,y,images,truth
        del bank
    print(f"Grad-CAM scope complete: {scope}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope",choices=("smoke",*SCOPES))
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Grad-CAM refits")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    cv2.setNumThreads(1)
    device=torch.device("cuda:0")
    trainer=Trainer(device,6)
    if args.scope=="smoke":
        for architecture in ("resnet50","vit_b_16","swin_t"):
            model=build_model(architecture,"scratch",6).to(device)
            image=torch.randint(0,256,(1,3,224,224),dtype=torch.uint8,device=device)
            scores,active,layer=cam(model,trainer,architecture,image,torch.zeros(1,dtype=torch.long,device=device))
            assert scores.shape==(1,224,224)
            print(f"Grad-CAM smoke {architecture}: layer={layer}, active={active[0]}",flush=True)
            del model
        return
    data=BenchmarkData()
    labels=data.labels.to(device)
    external="external" in args.scope
    secondary=secondary_inputs(data.classes,device) if external else None
    with threadpool_limits(limits=2):
        load_scope(args.scope,OUTPUT/args.scope,(data,labels,trainer,secondary))


if __name__=="__main__":
    main()
