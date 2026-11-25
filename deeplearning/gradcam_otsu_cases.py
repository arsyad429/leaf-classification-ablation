"""Refit and render matched-target Grad-CAM for selected external Otsu cases."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from data import BenchmarkData, ROOT, SEED, resize_for_model
from gradcam_refit import cam
from run_benchmark import atomic_write, training_data, write_csv
from run_external import secondary_inputs
from run_otsu import OtsuBank, otsu_rgb
from training import Trainer

RESULTS = ROOT / "deeplearning/results"
DEFAULT = RESULTS / "otsu/external_evaluation/eda_otsu_vit_swin/gradcam_cases"
GAIN = ("inat_19219491.jpg", "inat_245161059.jpg", "inat_418438754.jpg",
        "inat_245774771.jpg", "inat_471692010.jpg")
LOSS = ("inat_721118477.jpg", "inat_728790409.jpg", "inat_715480225.jpg", "inat_757327216.jpg")
MODELS = {"vit_b_16": "ViT-B/16", "swin_t": "Swin-T", "resnet50": "ResNet-50"}
PAKU = "Paku tanduk rusa"
BG = "#f7f9fc"
INK = "#162c42"
SUB = "#435c73"
BLUE = "#275e8b"
TEAL = "#137d70"
RUST = "#aa5136"


def load_archives():
    paths = {
        "N": RESULTS / "external_evaluation/runs.jsonl",
        "S-O": RESULTS / "otsu/external_evaluation/runs.jsonl",
    }
    matched = {}
    for condition, path in paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for run in rows:
            cfg = run["config"]
            if (cfg["input"] == condition and cfg["initialization"] == "pretrained"
                and not cfg["augmentation"] and cfg["architecture"] in MODELS):
                key = (cfg["architecture"], condition)
                if key in matched:
                    raise ValueError(f"Duplicate archive recipe: {key}")
                matched[key] = run
    if len(matched) != 6:
        raise ValueError("Expected six archived recipes")
    return matched, paths


def heading(fig, title, subtitle):
    fig.text(.055,.947,title,fontsize=19,fontweight="bold",va="top",color=INK)
    fig.text(.055,.884,subtitle,fontsize=10,va="top",color=SUB)


def save(fig, path):
    fig.savefig(path,dpi=165,facecolor=BG)
    plt.close(fig)


def overlay(rgb, mask, active):
    if not active:
        return rgb
    colors = cv2.applyColorMap(np.rint(mask*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1]
    alpha=(.66*mask)[...,None]
    return np.rint(rgb*(1-alpha)+colors*alpha).astype(np.uint8)


def draw_case(out, meta, rgb, panels, gain):
    photo_id = meta["photo_id"]
    label = meta["label"]
    keys = ([(m,c,"true") for m in ("vit_b_16","swin_t") for c in ("N","S-O")] if gain else
            [("resnet50","N","true"),("resnet50","S-O","true"),("resnet50","S-O","predicted")])
    fig,axes=plt.subplots(1,1+len(keys),figsize=(3.25*(1+len(keys)),6),facecolor=BG)
    fig.subplots_adjust(left=.055,right=.975,top=.755,bottom=.24,wspace=.19)
    title = "Otsu memperbaiki prediksi ViT dan Swin" if gain else "ResNet-50 kehilangan prediksi paku setelah Otsu"
    heading(fig,title,f"iNaturalist {photo_id} | Label asli: {label} | Pretrained, augmentasi OFF")
    axes[0].imshow(rgb)
    axes[0].set_title("Input uji bersama",fontweight="bold",fontsize=9)
    axes[0].axis("off")
    for ax,key in zip(axes[1:],keys):
        model,condition,target_kind=key
        panel=panels[key][meta["image"]]
        ax.imshow(overlay(rgb,panel["map"],panel["active"]))
        ax.axis("off")
        ax.set_title(f"{MODELS[model]} | {condition}",fontweight="bold",fontsize=9)
        note=f"Target: {panel['target']}\nPrediksi: {panel['prediction']}\nSkor target: {panel['score']:.3f}"
        if not panel["active"]:
            note+="\nCAM datar"
        ax.text(.5,-.075,note,transform=ax.transAxes,ha="center",va="top",fontsize=8,color=SUB)
    fig.text(.055,.072,"Grad-CAM dinormalisasi per panel. Skor target adalah softmax. PCD hanya pada foto latih.",fontsize=8.5,color=SUB)
    fig.text(.055,.035,f"Foto iNaturalist {photo_id} | {meta['license_code'].upper()} | Atribusi lengkap tersedia di atribusi_foto.csv.",
             fontsize=8.5,color=SUB)
    filename=f"{'gain_vit_swin' if gain else 'loss_resnet_paku'}_{photo_id}.png"
    save(fig,out/filename)


def frond_panel(out, primary):
    candidates=[]
    for i,image in enumerate(primary.photos):
        if primary.classes[int(primary.labels[i])]==PAKU:
            cut,mask,_=otsu_rgb(image,"S-O")
            candidates.append((float(mask.mean()),i,cut,mask))
    candidates.sort(key=lambda t:t[0])
    selected=[candidates[i] for i in (0,len(candidates)//2,len(candidates)-1)]
    fig,axes=plt.subplots(3,3,figsize=(10.3,10.2),facecolor=BG)
    fig.subplots_adjust(left=.095,right=.94,top=.80,bottom=.09,wspace=.20,hspace=.37)
    heading(fig,"Mask Otsu pada foto latih paku tanduk rusa",
            "Tiga contoh berdasarkan fraksi mask | Foto primer sebelum dan sesudah S-O")
    for j,(coverage,i,cut,mask) in enumerate(selected):
        binary=np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,axis=2)
        for ax,(name,array) in zip(axes[j],(("N | Input asli",primary.photos[i]),
                                           (f"Mask | {coverage:.1%}",binary),("S-O | Input latih",cut))):
            ax.imshow(resize_for_model(array).transpose(1,2,0))
            ax.set_title(name,fontsize=9)
            ax.axis("off")
        axes[j,0].text(.5,-.07,primary.ids[i],transform=axes[j,0].transAxes,ha="center",va="top",fontsize=8)
    fig.text(.095,.035,"Putih menunjukkan area yang dipertahankan. Ketepatan terhadap daun memerlukan mask anotasi.",fontsize=8.5,color=SUB)
    save(fig,out/"resnet_paku_mask_latih.png")


def refit(primary, labels, trainer, records, meta, secondary, test_y):
    index={r["image"]:i for i,r in enumerate(meta)}
    truth=[r["label"] for r in meta]
    panels={}
    evidence=[]
    agreement=[]
    paku=[]
    for condition in ("N","S-O"):
        bank=(primary.input_bank("N",labels.device) if condition=="N"
              else OtsuBank("S-O",primary.photos,labels.device))
        x,y=training_data(bank,labels,list(range(60)),False)
        for model in MODELS:
            run=records[model,condition]
            cfg=run["config"]
            lr,epochs=cfg["learning_rate"],cfg["epochs"]
            fitted,_=trainer.fit(model,"pretrained",lr,epochs,SEED,x,y)
            _,_,_,pred,probs=trainer.evaluate(fitted,secondary,test_y)
            old={r["image"]:r for r in run["predictions"]}
            if set(old)!=set(index):
                raise ValueError("Archive and evaluation image IDs differ")
            prediction=[primary.classes[p] for p in pred]
            matches=sum(old[r["image"]]["pred"]==p and old[r["image"]]["true"]==r["label"]
                        for r,p in zip(meta,prediction))
            if matches!=150:
                raise ValueError(f"Refit predictions differ from archive: {model}/{condition}, {matches}/150")
            agreement.append(dict(model=model,input=condition,seed=SEED,learning_rate=lr,epochs=epochs,matching_predictions=matches,photos=150))
            names=LOSS if model=="resnet50" else GAIN
            ids=[index[name] for name in names]
            kinds=("true","predicted") if (model,condition)==("resnet50","S-O") else ("true",)
            for target_kind in kinds:
                targets=test_y[ids] if target_kind=="true" else torch.tensor([pred[i] for i in ids],device=labels.device)
                maps,active,layer=cam(fitted,trainer,model,secondary[ids],targets)
                position=(model,condition,target_kind)
                panels[position]={}
                for k,(name,n,target) in enumerate(zip(names,ids,targets.tolist())):
                    right=prediction[n]==truth[n]
                    expected=(condition=="S-O" if model!="resnet50" else condition=="N")
                    if right!=expected:raise ValueError("Archived case no longer satisfies paired selection")
                    record=dict(image=name,observation_id=meta[n]["observation_id"],model=model,input=condition,
                                true=truth[n],prediction=prediction[n],target_kind=target_kind,target=primary.classes[target],
                                score=probs[n][target],active=active[k],layer=layer)
                    evidence.append(record)
                    panels[position][name]=dict(**record,map=maps[k])
            if model=="resnet50":
                paku_idx=primary.classes.index(PAKU)
                for n,label in enumerate(truth):
                    if label==PAKU:
                        paku.append(dict(input=condition,image=meta[n]["image"],true=label,prediction=prediction[n],paku_score=probs[n][paku_idx]))
                counts=Counter(prediction)
                print(f"ResNet {condition} predicted paku on {counts[PAKU]} of 150",flush=True)
            del fitted
            print(f"Grad-CAM case refit {model} {condition}: {matches}/150 matched",flush=True)
        del bank,x,y
    return panels,evidence,agreement,paku


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"run_config.json").exists():
        raise FileExistsError("The requested case Grad-CAM output already exists")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    cv2.setNumThreads(1)
    device=torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    primary=BenchmarkData()
    trainer=Trainer(device,len(primary.classes))
    labels=primary.labels.to(device)
    meta,secondary,test_y=secondary_inputs(primary.classes,device)
    records,sources=load_archives()
    with threadpool_limits(limits=2):
        panels,evidence,agreement,paku=refit(primary,labels,trainer,records,meta,secondary,test_y)
        frond_panel(args.output,primary)
    index={r["image"]:i for i,r in enumerate(meta)}
    credits=[]
    for name in GAIN+LOSS:
        i=index[name]
        row=meta[i]
        if row["license_code"] not in ("cc-by","cc0"):
            raise ValueError("Unexpected image license")
        draw_case(args.output,row,secondary[i].cpu().numpy().transpose(1,2,0),panels,name in GAIN)
        credits.append({k:row[k] for k in ("image","label","source_url","photo_url","license_code","license_url","attribution")})
    write_csv(args.output/"skor_dan_target_cam.csv",evidence)
    write_csv(args.output/"refits.csv",agreement)
    write_csv(args.output/"prediksi_paku_resnet.csv",paku)
    write_csv(args.output/"atribusi_foto.csv",credits)
    plan=dict(
        seed=SEED, pretrained=True, augmentation=False, scopes=["N","S-O"],
        target="true class for paired maps; S-O wrong class also shown for ResNet",
        photos="same 150 unmodified secondary images with common model tensor preparation",
        archive_hashes={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in sources.items()},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection={"shared_vit_swin_gains":GAIN,"resnet_paku_errors":LOSS},
    )
    atomic_write(args.output/"run_config.json",lambda stream:stream.write(json.dumps(plan,indent=2)+"\n"))
    print("Matched Grad-CAM cases complete",flush=True)


if __name__=="__main__":
    main()
