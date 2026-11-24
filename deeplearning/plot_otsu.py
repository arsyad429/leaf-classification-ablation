"""Build Otsu analysis figures from completed benchmark artifacts."""

import argparse
import csv
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from data import BenchmarkData, ROOT, preprocess, resize_for_model
from run_benchmark import write_csv
from run_otsu import CONDITIONS, COMPARISONS, DESTINATION, HISTORY, PRIMARY_HISTORY, otsu_rgb, read_rows

plt.rcParams.update({"figure.facecolor": "#f7f9fc", "savefig.facecolor": "#f7f9fc",
                     "font.family": "DejaVu Sans", "text.color": "#172a3d"})
MODELS = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "swin_t": "Swin-T"}
KEYS = ("input", "architecture", "initialization", "augmentation")
LABELS = ("Aglaonema", "Daun nangka", "Kamboja jepang", "Lidah metua",
          "Paku tanduk rusa", "Peperomium semangka")
SHORT = dict(zip(LABELS, ("Aglaonema", "Nangka", "Kamboja", "Lidah metua", "Paku tanduk rusa", "Peperomia")))
META = ROOT / "Dataset_Sekunder/metadata.csv"


def key(row):
    return tuple(str(row[k]) for k in KEYS)


def header(fig, title, context):
    fig.text(.06,.948,title,fontsize=19,fontweight="bold",va="top")
    fig.text(.06,.884,context,fontsize=10,va="top",color="#455b71")


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=155)
    plt.close(fig)


def group_predictions(rows, count):
    groups = defaultdict(dict)
    for row in rows:
        config = key(row)
        if row["image"] in groups[config]:
            raise ValueError("Duplicate prediction")
        groups[config][row["image"]] = row
    if any(len(g)!=count for g in groups.values()):
        raise ValueError("Incomplete predictions")
    return groups


def panel_confusion(folder, chosen, summaries, predictions, f1, external):
    condition, model, weights, aug = chosen
    references = COMPARISONS[condition]
    arms = (*references,condition)
    fig, axes = plt.subplots(1,len(arms),figsize=(5.2*len(arms)+1,6.1),squeeze=False)
    fig.subplots_adjust(left=.10,right=.98,top=.76,bottom=.23,wspace=.38)
    context = "150 foto sekunder yang sama, PCD hanya pada train" if external else "60 foto primer, prediksi OOF"
    header(fig,"Confusion matrix | Otsu dan pembanding",
           f"{MODELS[model]} | {weights} | Aug {'ON' if aug=='True' else 'OFF'} | {context}")
    for ax, arm in zip(axes[0],arms):
        cfg = (arm,model,weights,aug)
        cells = np.zeros((6,6),dtype=int)
        for row in predictions[cfg].values():
            cells[LABELS.index(row["true"]),LABELS.index(row["pred"])]+=1
        rate = cells/cells.sum(axis=1,keepdims=True)
        ax.imshow(rate,vmin=0,vmax=1,cmap="Blues")
        for i,j in np.ndindex(cells.shape):
            count=cells[i,j]
            ax.text(j,i,f"{count}\n{rate[i,j]:.0%}" if count else "·",ha="center",va="center",fontsize=8,
                    color="white" if rate[i,j]>.55 else "#172a3d")
        ax.set_title(f"{arm} | Macro-F1 {float(summaries[cfg][f1]):.3f}",fontsize=11)
        ax.set_xticks(range(6),SHORT.values(),rotation=40,ha="right",fontsize=8)
        ax.set_yticks(range(6),SHORT.values(),fontsize=8)
        ax.set_xlabel("Prediksi")
        ax.tick_params(length=0)
    axes[0,0].set_ylabel("Label sebenarnya")
    save(fig,folder,"01_confusion_matrix.png")


def plots_scores(folder, summaries, metric, external):
    arms = ("N","E","S-K","S-O","E-S-O")
    configs = list(product(MODELS,("pretrained","scratch"),("False","True")))
    matrix = np.array([[float(summaries[(arm, *cfg)][metric]) for arm in arms] for cfg in configs])
    delta = np.array([matrix[:,3]-matrix[:,0],matrix[:,3]-matrix[:,2],matrix[:,4]-matrix[:,1]]).T
    ylabels = [f"{MODELS[m]} | {'Pretrained' if w=='pretrained' else 'Acak'} | Aug {'ON' if a=='True' else 'OFF'}"
               for m,w,a in configs]
    for name, values, xlabels, context in (
        ("02_heatmap_macro_f1",matrix,arms,"Macro-F1 absolut"),
        ("03_selisih_macro_f1",delta,("S-O - N","S-O - S-K","E-S-O - E"),"Selisih macro-F1 pada konfigurasi setara"),
    ):
        fig,ax=plt.subplots(figsize=(11.8,7.8))
        fig.subplots_adjust(left=.29,right=.90,top=.82,bottom=.12)
        header(fig,context,"Evaluasi eksternal 150 foto | PCD hanya di train" if external else "Rata-rata 5 outer fold primer")
        if name.startswith("03"):
            limit=max(.01,float(np.abs(values).max()))
            im=ax.imshow(values,aspect="auto",cmap="PuOr",norm=TwoSlopeNorm(0,-limit,limit))
        else:
            im=ax.imshow(values,aspect="auto",cmap="Blues",vmin=0,vmax=1)
        for i,j in np.ndindex(values.shape):
            value=values[i,j]
            rgb=im.cmap(im.norm(value))[:3]
            ax.text(j,i,f"{value:+.3f}" if name.startswith("03") else f"{value:.3f}",ha="center",
                    va="center",fontsize=9,fontweight="bold",color="white" if sum(rgb)/3<.52 else "#172a3d")
        ax.set_xticks(range(len(xlabels)),xlabels)
        ax.set_yticks(range(12),ylabels,fontsize=8.5)
        ax.tick_params(length=0)
        for y in (3.5,7.5): ax.axhline(y,color="white",lw=2)
        fig.colorbar(im,ax=ax,shrink=.85,pad=.03)
        save(fig,folder,name+".png")


def photo_examples(folder, chosen, predictions, data, external):
    condition, model, weights, aug = chosen
    reference = COMPARISONS[condition][0]
    after = predictions[chosen]
    before = predictions[(reference,model,weights,aug)]
    if set(after)!=set(before): raise ValueError("Comparison has different photo IDs")
    meta = ({row["image"]:row for row in read_rows(META)} if external else
            {name: {"image":name,"label":LABELS[int(data.labels[i])]} for i,name in enumerate(data.ids)})
    file="04_foto_perubahan_prediksi.png"
    gallery=[]
    for better in (True,False):
        changed = [
            image for image in after
            if (after[image]["true"]==after[image]["pred"])==better
            and (before[image]["true"]==before[image]["pred"])!=better
        ]
        ranked = sorted((image for image in changed if not external or meta[image]["license_code"] in ("cc-by","cc0")),
                        key=lambda image:float(after[image]["p_"+after[image]["pred"]]),reverse=True)
        gallery.append((better,len(changed),ranked[:4]))

    fig,axes=plt.subplots(2,4,figsize=(15,8.6))
    fig.subplots_adjust(left=.06,right=.96,top=.76,bottom=.13,hspace=.72,wspace=.28)
    header(fig,"Foto yang prediksinya berubah dengan Otsu",
           f"{MODELS[model]} | {weights} | Aug {'ON' if aug=='True' else 'OFF'} | {reference} ke {condition}")
    credits=[]
    for y,(became_correct,count,examples) in enumerate(gallery):
        figs=axes[y]
        for ax in figs: ax.axis("off")
        figs[0].set_title(f"{'Jadi benar' if became_correct else 'Jadi salah'} | {count} foto",
                          fontweight="bold",fontsize=10,loc="left",pad=10)
        if not examples:
            figs[0].text(.5,.5,"Tidak ada contoh",ha="center",va="center")
        for ax,image in zip(figs,examples):
            row=meta[image]
            if external:
                path=ROOT/"Dataset_Sekunder"/row["label"]/image
                with Image.open(path) as source:
                    ax.imshow(source.convert("RGB"))
                credits.append({k:row[k] for k in ("image","label","source_url","photo_url","license_code","license_url","attribution")})
            else:
                ax.imshow(data.photos[data.ids.index(image)])
            old,new=SHORT[before[image]["pred"]],SHORT[after[image]["pred"]]
            caption=f"Label: {SHORT[row['label']]}\n{reference}: {old} | {condition}: {new}"
            if external:
                caption+=f"\niNat {row['photo_id']} | {row['license_code'].upper()}"
            else:
                caption+="\n"+"\n".join(textwrap.wrap(image,28))
            ax.text(.5,-.065,caption,transform=ax.transAxes,ha="center",va="top",fontsize=8)
    fig.text(.06,.045,"Foto sekunder ditampilkan apa adanya. Atribusi lengkap ada di CSV." if external
             else "Perbandingan memakai prediksi OOF untuk foto primer yang sama.",fontsize=8.5,color="#455b71")
    save(fig,folder,file)
    write_csv(folder/"atribusi_foto.csv",credits)


def plot_loss(folder, chosen, folder_results, external):
    name="runs.jsonl" if external else "folds.jsonl"
    records=[json.loads(line) for line in (folder_results/name).read_text().splitlines()]
    matching=[r for r in records if key(r["config"])==chosen]
    fig,ax=plt.subplots(figsize=(10.8,6))
    fig.subplots_adjust(left=.11,right=.96,top=.76,bottom=.14)
    condition,model,weights,aug=chosen
    description=f"{condition} | {MODELS[model]} | {weights} | Aug {'ON' if aug=='True' else 'OFF'}"
    if external:
        curve=matching[0]["training_curve"]
        ax.plot([r["epoch"] for r in curve],[r["train_loss"] for r in curve],color="#1a6489",label="Train",marker="o",ms=3)
        header(fig,"Kurva loss fit final",description+" | Seluruh 60 foto primer")
    else:
        fold=next(r for r in matching if r["fold"]==0)
        for trial,color in zip((r for r in fold["inner"] if r["learning_rate"]==fold["selected_lr"]),
                               ("#1a6489","#b56525","#337970")):
            xs=[r["epoch"] for r in trial["curve"]]
            n=trial["inner_fold"]+1
            ax.plot(xs,[r["train_loss"] for r in trial["curve"]],color=color,label=f"Inner {n} train")
            ax.plot(xs,[r["val_loss"] for r in trial["curve"]],color=color,ls="--",label=f"Inner {n} validasi")
        header(fig,"Kurva loss inner CV",description+f" | Outer fold 1 | LR {fold['selected_lr']:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(axis="y",color="#e0e7ee")
    ax.legend(frameon=False,ncol=2)
    save(fig,folder,"05_kurva_loss.png")


def mask_panel(folder,data):
    with threadpool_limits(limits=2):
        masks = {(condition,index):otsu_rgb(rgb,condition)
                 for condition,index in product(CONDITIONS,range(len(data.ids)))
                 for rgb in [data.photos[index]]}
    stats=[dict(input=condition,image=data.ids[index],label=LABELS[int(data.labels[index])],
                otsu_threshold=threshold,mask_fraction=float(mask.mean()))
           for (condition,index),(_,mask,threshold) in masks.items()]
    write_csv(folder.parent/"mask_audit.csv",stats)
    fig,axes=plt.subplots(6,5,figsize=(13.5,15))
    fig.subplots_adjust(left=.13,right=.97,top=.82,bottom=.08,hspace=.37,wspace=.20)
    header(fig,"Otsu pada foto latih primer","Satu foto per kelas dipilih berdasarkan fraksi mask S-O terkecil di kelasnya")
    for y,label in enumerate(LABELS):
        ids=[i for i in range(len(data.ids)) if LABELS[int(data.labels[i])]==label]
        index=min(ids,key=lambda i:masks["S-O",i][1].mean())
        rgb=data.photos[index]
        so,mask1,_=masks["S-O",index]
        eso,mask2,_=masks["E-S-O",index]
        rgbmask=lambda mask:np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,axis=2)
        columns=[("N",rgb),(f"Mask S-O {mask1.mean():.0%}",rgbmask(mask1)),("S-O",so),
                 (f"Mask E-S-O {mask2.mean():.0%}",rgbmask(mask2)),("E-S-O",eso)]
        for ax,(caption,array) in zip(axes[y],columns):
            ax.imshow(resize_for_model(array).transpose(1,2,0))
            ax.set_title(caption,fontsize=9)
            ax.axis("off")
        bbox=axes[y,0].get_position()
        fig.text(.10,(bbox.y0+bbox.y1)/2,"\n".join(textwrap.wrap(label,12)),ha="right",va="center",fontsize=9)
    save(fig,folder,"06_panel_otsu_training.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("benchmark","external_evaluation"))
    args=parser.parse_args()
    external=args.stage=="external_evaluation"
    output=DESTINATION/args.stage
    folder=output/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    baseline=HISTORY if external else PRIMARY_HISTORY
    score_name="macro_f1" if external else "macro_f1_mean"
    csv_predictions="predictions.csv" if external else "oof_predictions.csv"
    scores=read_rows(output/"summary.csv")
    if len(scores)!=24:raise ValueError("Wait until all Otsu configurations finish")
    chosen=key(max(scores,key=lambda row:float(row[score_name])))
    combined={key(row):row for row in read_rows(baseline/"summary.csv")+scores}
    all_predictions=group_predictions(read_rows(baseline/csv_predictions)+read_rows(output/csv_predictions),
                                      150 if external else 60)
    data=BenchmarkData() if not external else None
    panel_confusion(folder,chosen,combined,all_predictions,score_name,external)
    plots_scores(folder,combined,score_name,external)
    photo_examples(folder,chosen,all_predictions,data,external)
    plot_loss(folder,chosen,output,external)
    if not external:
        cv2.setNumThreads(1)
        mask_panel(folder,data)
    print(f"Otsu figures {args.stage} complete",flush=True)


if __name__=="__main__":
    main()
