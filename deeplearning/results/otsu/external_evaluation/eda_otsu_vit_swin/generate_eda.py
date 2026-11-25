"""Build paired EDA for Otsu on pretrained ViT-B/16 and Swin-T."""

import argparse
from collections import Counter
import csv
import hashlib
import io
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[4]
MODELS = {"vit_b_16":"ViT-B/16", "swin_t":"Swin-T"}
LABELS = ("Aglaonema","Daun nangka","Kamboja jepang","Lidah metua","Paku tanduk rusa","Peperomium semangka")
SHORT = dict(zip(LABELS, ("Aglaonema","Nangka","Kamboja","Lidah metua","Paku","Peperomia")))
FIG_BG="#f7f9fc"
INK="#182c40"
MUTED="#435c73"
BLUE="#255d91"
TEAL="#147f75"
RUST="#b85535"


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lookup(rows,model,aug,condition):
    data={r["image"]:r for r in rows if
          (r["input"],r["architecture"],r["initialization"],r["augmentation"])==
          (condition,model,"pretrained",str(aug))}
    if len(data)!=150:
        raise ValueError(f"Expected 150 unique photos for {model}/{condition}/{aug}")
    return data


def paired(base,otsu):
    photos=[]
    summary=[]
    for model in MODELS:
        for aug in (False,True):
            before=lookup(base,model,aug,"N")
            after=lookup(otsu,model,aug,"S-O")
            if before.keys()!=after.keys():raise ValueError("Paired image IDs differ")
            batch=[]
            for image in sorted(before):
                a,b=before[image],after[image]
                label=a["true"]
                if b["true"]!=label:raise ValueError(f"Label mismatch: {image}")
                ok_n,ok_o=a["pred"]==label,b["pred"]==label
                change=("improved" if ok_o else "regressed") if ok_n!=ok_o else ("both_correct" if ok_n else "both_wrong")
                batch.append(dict(model=model,augmentation=aug,image=image,observation_id=a["observation_id"],
                                  true=label,pred_n=a["pred"],pred_otsu=b["pred"],change=change,
                                  true_score_n=a[f"p_{label}"],true_score_otsu=b[f"p_{label}"]))
            photos.extend(batch)
            for label in LABELS:
                values=[r for r in batch if r["true"]==label]
                assert len(values)==25
                n=sum(r["pred_n"]==label for r in values)
                o=sum(r["pred_otsu"]==label for r in values)
                p_n=sum(r["pred_n"]==label for r in batch)
                p_o=sum(r["pred_otsu"]==label for r in batch)
                summary.append(dict(model=model,augmentation=aug,label=label,photos=25,
                    correct_n=n,correct_otsu=o,recall_n=n/25,recall_otsu=o/25,
                    improved=sum(r["change"]=="improved" for r in values),
                    regressed=sum(r["change"]=="regressed" for r in values),
                    predicted_n=p_n,predicted_otsu=p_o,false_positive_n=p_n-n,false_positive_otsu=p_o-o))
    return photos,summary


def head(fig,title,note):
    fig.text(.062,.947,title,fontsize=20,va="top",fontweight="bold")
    fig.text(.062,.889,note,fontsize=10,color=MUTED,va="top")


def foot(fig,note):
    fig.text(.062,.035,note,fontsize=9,color=MUTED)


def save(fig,path):
    fig.savefig(path,dpi=160)
    plt.close(fig)


def recall_figure(out, summary):
    fig,axes=plt.subplots(1,2,figsize=(14.2,7.3),sharey=True)
    fig.subplots_adjust(left=.16,right=.95,top=.76,bottom=.15,wspace=.20)
    head(fig,"Otsu memperbaiki pengenalan kamboja dan aglaonema",
         "ViT-B/16 dan Swin-T pretrained | Augmentasi OFF | 25 foto sekunder per kelas")
    for ax,model in zip(axes,MODELS):
        data=[r for r in summary if r["model"]==model and not r["augmentation"]]
        y=np.arange(len(LABELS))
        a=[r["correct_n"] for r in data];b=[r["correct_otsu"] for r in data]
        ax.barh(y-.16,a,.28,label="N",color=BLUE)
        ax.barh(y+.16,b,.28,label="S-O",color=TEAL)
        for idx,left,right in zip(y,a,b):
            ax.text(left+.26,idx-.16,str(left),fontsize=9,va="center",color=BLUE)
            ax.text(right+.26,idx+.16,str(right),fontsize=9,va="center",color=TEAL)
        ax.set_yticks(y,LABELS,fontsize=8.5)
        ax.set_ylim(5.55,-.55)
        ax.set_xlim(0,27)
        ax.set_xticks([0,5,10,15,20,25])
        ax.set_xlabel("Prediksi benar dari 25")
        ax.set_title(MODELS[model],loc="left",fontsize=12,fontweight="bold")
        ax.grid(axis="x",color="#dce4ec",linewidth=.8)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right",frameon=False,ncol=2)
    foot(fig,"PCD hanya dipakai untuk foto latih. Tolok ukur tiap pasangan adalah foto sekunder yang sama.")
    save(fig,out/"01_recall_per_kelas.png")


def mask_figure(out,masks):
    fig,ax=plt.subplots(figsize=(11.8,6.8))
    fig.subplots_adjust(left=.20,right=.94,top=.77,bottom=.16)
    head(fig,"Berapa banyak piksel latih dipertahankan Otsu?",
         "Audit 60 foto primer | S-O memilih ambang per foto dari kanal a* Lab")
    so=[r for r in masks if r["input"]=="S-O"]
    assert len(so)==60
    for i,label in enumerate(LABELS):
        values=sorted(float(r["mask_fraction"])*100 for r in so if r["label"]==label)
        y=np.full(len(values),i)+np.linspace(-.13,.13,len(values))
        ax.scatter(values,y,s=36,color=BLUE,alpha=.67,label="Satu foto" if i==0 else None,zorder=3)
        md=float(np.median(values))
        ax.plot([md,md],[i-.23,i+.23],color=TEAL,lw=3,label="Median kelas" if i==0 else None,zorder=4)
        ax.text(99,i,f"{md:.1f}%",va="center",ha="right",fontsize=9,color=TEAL)
    ax.set_xlim(0,101)
    ax.set_ylim(5.55,-.55)
    ax.set_yticks(range(6),LABELS,fontsize=10)
    ax.set_xticks(range(0,101,20),[f"{n}%" for n in range(0,101,20)])
    ax.set_xlabel("Fraksi piksel dalam mask")
    ax.set_axisbelow(True)
    ax.grid(axis="x",color="#e0e6ec")
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.52,1.1))
    foot(fig,"Fraksi mask mengukur luas area yang dipertahankan. Lokasi daun sebenarnya tidak dianotasi.")
    save(fig,out/"02_fraksi_mask_otsu_primer.png")


def plot_examples(out, photos, manifest, dataset_zip):
    anchor={(r["model"],r["image"]):r for r in photos if not r["augmentation"]}
    examples=[
        "inat_19219491.jpg", "inat_245161059.jpg", "inat_418438754.jpg",
        "inat_245774771.jpg", "inat_471692010.jpg", "inat_755244004.jpg",
    ]
    fig,axes=plt.subplots(2,3,figsize=(13.9,9.4))
    fig.subplots_adjust(left=.06,right=.94,top=.79,bottom=.135,hspace=.68,wspace=.30)
    head(fig,"Foto yang berubah pada ViT dan Swin setelah training Otsu",
         "Lima contoh membaik dan satu memburuk | Bobot pretrained dan augmentasi OFF")
    credits=[]
    with zipfile.ZipFile(dataset_zip) as z:
        for i,(ax,image) in enumerate(zip(axes.flat,examples)):
            data=manifest[image]
            label=data["label"]
            a,b=(anchor[m,image] for m in MODELS)
            expected="improved" if i<5 else "regressed"
            assert a["change"]==b["change"]==expected
            assert data["license_code"] in ("cc-by","cc0")
            raw=z.read(f"Dataset_Sekunder/{label}/{image}")
            assert hashlib.sha256(raw).hexdigest()==data["sha256"]
            with Image.open(io.BytesIO(raw)) as source:
                ax.imshow(source.convert("RGB"))
            ax.axis("off")
            color=TEAL if expected=="improved" else RUST
            status="Membaik" if expected=="improved" else "Memburuk"
            ax.set_title(f"{status} | {label}",fontsize=10,fontweight="bold",color=color,pad=8)
            author=data["attribution"].split(", some rights reserved")[0].split(", no rights reserved")[0]
            author=author.removeprefix("(c) ").strip()
            if not author or "no rights reserved" in author.lower(): author="@"+data["observer"]
            text=f"ViT: {SHORT[a['pred_n']]} → {SHORT[a['pred_otsu']]}\nSwin: {SHORT[b['pred_n']]} → {SHORT[b['pred_otsu']]}\n"
            text+="\n".join(textwrap.wrap(f"{author} | {data['license_code'].upper()} | iNat {data['photo_id']}",39))
            ax.text(.5,-.07,text,ha="center",va="top",transform=ax.transAxes,color=MUTED,fontsize=8,linespacing=1.27)
            credits.append({k:data[k] for k in ("image","label","source_url","photo_url","license_code","license_url","attribution")})
    foot(fig,"Foto uji tampil tanpa PCD. Panah menunjukkan prediksi N → S-O. Sumber per foto tersedia di atribusi_foto.csv.")
    save(fig,out/"03_galeri_perubahan_prediksi.png")
    write(out/"atribusi_foto.csv",credits)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip",type=Path,required=True,help="Local ZIP with the 150 secondary photos")
    parser.add_argument("--repo",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=FOLDER)
    args=parser.parse_args()
    paths=args.repo/"deeplearning/results"
    otsu=paths/"otsu"
    args.output.mkdir(parents=True,exist_ok=True)
    font_path=Path("C:/Windows/Fonts/msjh.ttc")
    families=["DejaVu Sans"]
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        families.append("Microsoft JhengHei")
    plt.rcParams.update({"font.family":families,"figure.facecolor":FIG_BG,"savefig.facecolor":FIG_BG,"text.color":INK,
                         "axes.spines.top":False,"axes.spines.right":False,"xtick.color":MUTED,"ytick.color":MUTED})
    base=read(paths/"external_evaluation/predictions.csv")
    new=read(otsu/"external_evaluation/predictions.csv")
    meta={r["image"]:r for r in read(args.repo/"Dataset_Sekunder/metadata.csv")}
    photos,summary=paired(base,new)
    assert len(photos)==600 and len(summary)==24
    write(args.output/"perbandingan_per_foto.csv",photos)
    write(args.output/"ringkasan_per_kelas.csv",summary)
    recall_figure(args.output,summary)
    mask_figure(args.output,read(otsu/"benchmark/mask_audit.csv"))
    plot_examples(args.output,photos,meta,args.dataset_zip)
    print("Verified pairs and generated three EDA figures for 150 external photos")


if __name__=="__main__":
    main()
