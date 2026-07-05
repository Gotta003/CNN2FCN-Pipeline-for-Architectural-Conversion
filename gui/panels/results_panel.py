from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import customtkinter as ctk
import tkinter as tk
from gui.state import AppState
from gui.panels.base_panel import BasePanel
from src.assets import Icons
import threading
from typing import Optional, Any

HAS_MPL=None

def _ensure_mpl():
    global HAS_MPL
    if HAS_MPL is not None:
        return HAS_MPL
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        HAS_MPL=True
    except ImportError:
        HAS_MPL=False
    return HAS_MPL
    
_PTH_META: dict[str, tuple[str, str]]={
    "kws_multi_cnn_model_pytorch": ("Teacher CNN", "teacher"),
    "refiner": ("ENFORCE Refiner", "enforce"),
    "bridge_stage1": ("Bridge S1 (Hint)", "hint"),
    "bridge_stage2": ("Bridge S2 (DKD)", "dkd"),
    "anchor_1M": ("Anchor 1M", "anchor"),
}

_CHART_BG="#1A1A1A"
_COLORS=["#378ADD", "#1D9E75", "#EF9F27", "#E24B4A", "#7F77DD", "#D85A30", "#009688"]

def _style_ax(ax):
    ax.set_facecolor(_CHART_BG)
    ax.tick_params(colors="gray", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.title.set_color("gray")
    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    
def _embed_fig(parent, fig)->FigureCanvasTkAgg:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas=FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas

class TrainingCurvesView(ctk.CTkFrame):
    def __init__(self, master, metrics: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        if not _ensure_mpl():
            ctk.CTkLabel(self, text="matplotlib not available").pack()
            return
        if not metrics:
            ctk.CTkLabel(self, text="No training metrics found.\nRun the pipeline first - each stage writes\na metrics_{stage}.json sidecar.", text_color=("gray50", "gray55"), font=ctk.CTkFont(size=12)).pack(expand=True)
            return
        self._render(metrics)
        
    def _render(self, m: dict):
        import matplotlib.pyplot as plt
        epochs=m.get("epochs", [])
        fig, axes=plt.subplots(1, 3, figsize=(13, 3.2), facecolor=_CHART_BG)
        fig.subplots_adjust(left=0.07, right=0.07, top=0.88, bottom=0.15, wspace=0.35)
        specs=[
            ("F1 Score", "train_f1", "val_f1", "#378ADD", "#1D9E75"),
            ("Loss", "train_loss", "val_loss", "#EF9F27", "#E24B4A"),
            ("Accuracy", "train_acc", "val_acc", "#7F77DD","#009688")
        ]
        for ax, (title, tr_key, va_key, c_tr, c_va) in zip(axes, specs):
            tr=m.get(tr_key, [])
            va=m.get(va_key, [])
            x=epochs if epochs else list(range(1, max(len(tr), len(va))+1))
            if tr:
                ax.plot(x[:len(tr)], tr, color=c_tr, lw=1.5, label="Train")
            if va:
                ax.plot(x[:len(va)], va, color=c_va, lw=1.5, label="Val", linestyle="--")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, facecolor="#2A2A2A", labelcolor="gray")
            _style_ax(ax)
        _embed_fig(self, fig)
        
class ConfusionMatrixView(ctk.CTkFrame):
    def __init__(self, master, cm, class_names, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        if not HAS_MPL or cm is None:
            ctk.CTkLabel(self, text="Run evaluation to generate confusion matrix.", text_color=("gray50", "gray55")).pack(expand=True)
            return
        self._render(cm, class_names)
        
    def _render(self, cm, class_names):
        import numpy as np
        import matplotlib.pyplot as plt
        n=len(class_names)
        fig, ax=plt.subplots(figsize=(max(5, n*0.65), max(4, n*0.6)), facecolor=_CHART_BG)
        row_sum=cm.sum(axis=1, keepdims=True).clip(1)
        cm_norm=cm/row_sum
        im=ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8, color="gray")
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names, fontsize=8, color="gray")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7, color="white" if cm_norm[i, j]>0.5 else "#AAA")
        ax.set_xlabel("Predicted", color="gray", fontsize=9)
        ax.set_ylabel("True", color="gray", fontsize=9)
        ax.set_title("Confusion Matrix", color="gray", fontsize=10)
        ax.set_facecolor(_CHART_BG)
        fig.patch.set_facecolor(_CHART_BG)
        plt.colorbar(im, ax=ax).ax.tick_params(colors="gray", labelsize=7)
        fig.tight_layout()
        _embed_fig(self, fig)
        
class PerClassF1View(ctk.CTkFrame):
    def __init__(self, master, class_names, per_class_metrics, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        if not HAS_MPL or per_class_metrics is None:
            ctk.CTkLabel(self, text="Run evaluation to generate per-class F1", text_color=("gray50", "gray55")).pack(expand=True)
            return
        self._render(class_names, per_class_metrics)
        
    def _render(self, class_names, pcm):
        import numpy as np
        import matplotlib.pyplot as plt
        n=len(class_names)
        f1s=[pcm.get(c, {}).get("f1", 0.0) for c in class_names]
        pre=[pcm.get(c, {}).get("precision", 0.0) for c in class_names]
        rec=[pcm.get(c, {}).get("recall", 0.0) for c in class_names]
        x=np.arange(n)
        bw=0.26
        fig, ax=plt.subplots(figsize=(max(7, n*0.7), 3.5), facecolor=_CHART_BG)
        ax.bar(x-bw, pre, bw, label="Precision", color="#378ADD", alpha=0.85)
        ax.bar(x, rec, bw, label="Recall", color="#1D9E75", alpha=0.85)
        ax.bar(x+bw, f1s, bw, label="F1", color="#EF9F27", alpha=0.85)
        best_idx=int(np.argmax(f1s))
        ax.annotate("best", xy=(x[best_idx]+bw, f1s[best_idx]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color="#EF9F27")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8, color="gray")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", color="gray")
        ax.set_title("Pre-class Precision / Recall / F1", color="gray", fontsize=10)
        ax.legend(fontsize=8, facecolor="#2A2A2A", labelcolor="gray")
        _style_ax(ax)
        fig.tight_layout()
        _embed_fig(self, fig)
        
class ClassMetricsTable(ctk.CTkScrollableFrame):
    _COLS=["Class", "Precision", "Recall", "F1"]
    _WIDTHS=[140, 100, 100, 100]
    
    def __init__(self, master, class_names, per_class_metrics, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(label_fg_color="transparent")
        #Header
        header=ctk.CTkFrame(self, fg_color=("gray82", "gray22"))
        header.pack(fill="x", padx=4, pady=(4, 0))
        for c, w in zip(self._COLS, self._WIDTHS):
            ctk.CTkLabel(header, text=c, width=w, anchor="center", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=2)
        if per_class_metrics is None:
            ctk.CTkLabel(self, text="Run evaluation first.", text_color=("gray50", "gray55")).pack(pady=12)
            return
        
        odd=False
        for cls in class_names:
            pcm=per_class_metrics.get(cls, {})
            pre=pcm.get("precision", float("nan"))
            rec=pcm.get("recall", float("nan"))
            f1=pcm.get("f1", float("nan"))
            try:
                f1c="#1D9E75" if f1>=0.80 else ("#EF9F27" if f1>=0.55 else "#E24B4A")
            except Exception:
                f1c=("gray50", "gray55")
            row_bg=("gray90", "gray18") if odd else ("gray94", "gray15")
            row=ctk.CTkFrame(self, fg_color=row_bg)
            row.pack(fill="x", pady=1)
            for val, w, c in zip([cls, f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}"], self._WIDTHS, [None, None, None, f1c],):
                ctk.CTkLabel(row, text=val, width=w, anchor="center", font=ctk.CTkFont(size=11), text_color=c if c else ("gray10", "gray90")).pack(side="left", padx=2)
            odd=not odd

class SummaryLog(ctk.CTkFrame):
    def __init__(self, master, summary: dict, class_names: list, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._build(summary, class_names)
        
    def _build(self, s: dict, class_names: list):
        txt=tk.Text(self, wrap="word", font=("Courier New", 11), bg="#1A1A1A", fg="#D4D4D4", relief="flat", padx=14, pady=10, state="disabled")
        sb=ctk.CTkScrollbar(self, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        #Colour tags
        txt.tag_configure("head", foreground="#C8A45A", font=("Courier New", 11, "bold"))
        txt.tag_configure("good", foreground="#1D9E75")
        txt.tag_configure("mid", foreground="#EF9F27")
        txt.tag_configure("bad", foreground="#E24B4A")
        txt.tag_configure("label", foreground="#888")
        txt.tag_configure("sep", foreground="#333")
        
        def w(text, tag=""):
            txt.configure(state="normal")
            txt.insert("end", text, tag)
            txt.configure(state="disabled")
            
        def val_tag(v, good=0.80, mid=0.55):
            try:
                return "good" if v>=good else ("mid" if v>=mid else "bad")
            except Exception:
                return "label"
            
        w("-- Global metrics ------------------------------\n", "head")
        for key, label in [
            ("final_val_f1", "Final Val F1"),
            ("best_val_f1", "Best Val F1"),
            ("final_train_f1", "Final Train F1"),
            ("final_val_acc", "Final Val Acc"),
            ("final_train_acc", "Final Train Acc"),
            ("final_val_loss", "Final Val Loss"),
            ("final_train_loss", "Final Train Loss"),
        ]:
            v=s.get(key)
            if v is None:
                continue
            w(f"    {label:<22}", "label")
            if "loss" in key:
                tag=val_tag(1-v)
            else:
                tag=val_tag(v)
            w(f"{v:.4f}\n", tag)
        
        w("\n-- Per-class metrics -------------------------\n", "head")
        pcm=s.get("per_class", {})
        if pcm:
          w(f"    {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}\n", "label")
          w("  " + "-"*50 + "\n", "sep")
          for cls in class_names:
              m=pcm.get(cls, {})
              pre=m.get("precision", float("nan"))
              rec=m.get("recall", float("nan"))
              f1=m.get("f1", float("nan"))
              w(f"  {cls:<16}", "label")
              w(f"{pre:>10.3f}", val_tag(pre))
              w(f"{rec:>10.3f}", val_tag(rec))
              w(f"{f1:>10.3f}\n", val_tag(f1))
        else:
            w(" (run evaluation to populate)\n", "label")  

class ModelMetricsView(ctk.CTkFrame):
    def __init__(self, master, model_path: Path, state: AppState, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._path=model_path
        self._state=state
        self._summary: dict={}
        self._cm=None
        self._pcm: dict={}
        self._class_names: list[str]=[]
        self._tabs=ctk.CTkTabview(self, anchor="nw")
        self._tabs.pack(fill="both", expand=True)
        for t in ["Training Curves", "Confusion Matrix", "Per-Class F1", "Class Metrics", "Summary Log"]:
            self._tabs.add(t)
        self._status=ctk.CTkLabel(self, text="Click 'Run evaluation' to load metrics.", font=ctk.CTkFont(size=11), text_color=("gray50", "gray55"))
        self._status.pack(pady=4)
        
        btn_row=ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(0, 6))
        self._eval_btn=ctk.CTkButton(btn_row, text="Run evaluation", image=Icons.run, width=160, command=self._run_eval)
        self._eval_btn.pack(side="left", padx=(0, 8))
        self._spinner=ctk.CTkLabel(btn_row, text="", font=ctk.CTkFont(size=11), text_color=("gray50", "gray55"))
        self._spinner.pack(side="left")
        self._load_training_curves()
        
    def _load_training_curves(self):
        stem=self._path.stem
        meta=_PTH_META.get(stem, ("", stem))
        stage=meta[1]
        loaded=False
        for fname in [f"metrics_{stage}.json", f"metrics_{stem}.json"]:
            sidecar=self._state.project_root / fname
            if sidecar.exists():
                try:
                    with open(sidecar) as f:
                        metrics=json.load(f)
                    self._summary.update(metrics.get("summary", {}))
                    frame=self._tabs.tab("Training Curves")
                    for w in frame.winfo_children():
                        w.destroy()
                    TrainingCurvesView(frame, metrics).pack(fill="both", expand=True)
                    loaded=True
                    break
                except Exception:
                    pass
        if not loaded:
            frame=self._tabs.tab("Training Curves")
            TrainingCurvesView(frame, {}).pack(fill="both", expand=True)
        
    def _run_eval(self):
        self._eval_btn.configure(state="disabled")
        self._spinner.configure(text="Running inference...")
        self._status.configure(text="")
        t=threading.Thread(target=self._eval_worker, daemon=True)
        t.start()
        
    def _eval_worker(self):
        try:
            result=self._do_eval()
            self.after(0, self._on_eval_done, result, None)
        except Exception as e:
            self.after(0, self._on_eval_done, None, str(e))
            
    def _do_eval(self) -> dict:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score)
        torch.backends.cudnn.enabled = False
        root=self._state.project_root
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "pipeline" / "stages"))
        cfg=self._state.config_data
        if not cfg:
            raise RuntimeError("No config loaded. Set project root first.")
        
        print("\n"+"="*70)
        print("[EVAL] _do_eval()")
        print(f"[EVAL] Checkpoint: {self._path}")
        print(f"[EVAL] Checkpoint stem: {self._path.stem}")
        from src.pipeline.data_utils import load_splits, DatasetNotPreparedError
        try:
            splits=load_splits(cfg, root)
        except DatasetNotPreparedError as e:
            raise RuntimeError(str(e))
        class_names=cfg["data"]["class_names"]
        n_classes=splits["n_classes"]
        UNKNOWN_SENTINEL=splits["unknown_sentinel"]
        known_indices=splits["known_indices"]
        self._class_names=class_names
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stem=self._path.stem
        #Val
        X_val=torch.cat([splits["X_val_spatial_k"], splits["X_val_spatial_u"]])
        y_val_unk=torch.full((splits["X_val_spatial_u"].size(0),), UNKNOWN_SENTINEL, dtype=torch.long)
        y_val=torch.cat([splits["y_val"], y_val_unk], dim=0)
        #Test
        X_test=torch.cat([splits["X_test_spatial_k"], splits["X_test_spatial_u"]])
        y_test_unk=torch.full((splits["X_test_spatial_u"].size(0),), UNKNOWN_SENTINEL, dtype=torch.long)
        y_test=torch.cat([splits["y_test"], y_test_unk])
        
        print(f"[EVAL] n_classes (da splits)  : {n_classes}")
        print(f"[EVAL] len(class_names)       : {len(class_names)}  "
              f"{'<<< MISMATCH with n_classes!' if len(class_names) != n_classes else '(ok)'}")
        print(f"[EVAL] known_indices          : {known_indices}")
        print(f"[EVAL] X_val shape: {tuple(X_val.shape)}")
        print(f"[EVAL] y_val shape: min={y_val.min().item()} max={y_val.max().item()} unique={sorted(y_val.unique().tolist())}")
        print(f"[EVAL] X_test shape: {tuple(X_test.shape)}")
        print(f"[EVAL] y_test shape: min={y_test.min().item()} max={y_test.max().item()} unique={sorted(y_test.unique().tolist())}")
        from src.pipeline.models import (TeacherCNN, LogitRefiner, HSRBridge, CompressedHSRBridge)
        s1_cfg=cfg.get("stage1", {})
        s2_cfg=cfg.get("stage2", {})
        nas_cfg=cfg.get("nas", {})
        td=cfg["data"]["tracker_dim"]
        if stem=="kws_multi_cnn_model_pytorch":
            print("[EVAL] Teacher CNN")
            model=TeacherCNN(input_shape=(1, 40, 40), num_classes=n_classes).to(device)
            print(f"[EVAL] Weights from: {self._path}")
            model.load_state_dict(torch.load(self._path, map_location=device))
            model.eval()
            print("[EVAL] load_state_dict: OK")
            def infer(xf):
                return model(xf.to(device))
            with torch.no_grad():
                sample_x=splits["X_test_spatial_k"][:512]
                sample_y=splits["y_test"][:512]
                samples_logits=infer(sample_x)
                sample_preds=samples_logits.argmax(1).cpu()
                quick_acc=(sample_preds==sample_y).float().mean().item()
                print(f"[EVAL] accuracy: {quick_acc*100:.2f}%")
        elif stem=="kws_multi_cnn_model_pytorch_energy":
            pass      
        elif stem=="refiner":
            raise RuntimeError("The ENFORCE Refiner maps logits to logits. Select a full model for evaluation")
        elif stem in ("bridge_stage1", "bridge_stage2"):
            teacher=TeacherCNN(input_shape=(1, 40, 40), num_classes=n_classes).to(device)
            teacher.load_state_dict(torch.load(root / cfg["teacher"]["weights_path"], map_location=device))
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad=False
            bridge=HSRBridge(teacher, input_shape=(1, 40, 40), tracker_dim=td, embed_dim=s1_cfg.get("embed_dim", 128), ranks=s1_cfg.get("hsr_ranks", [16, 8, 4]), dropout=s1_cfg.get("dropout", 0.4)).to(device)
            import torch.nn as nn
            bridge.classifier=nn.Linear(s1_cfg.get("embed_dim", 128), n_classes).to(device)
            bridge.load_state_dict(torch.load(self._path, map_location=device))
            bridge.eval()
            for p in bridge.parameters():
                p.requires_grad=False
                
            def infer(xf):
                out, _=bridge.forward_flat(xf.to(device))
                return out
        else:
            nas_json=root / nas_cfg.get("results_path", "nas_results.json")
            budget=None
            arch=None
            if "nas_bridge_" in stem:
                try:
                    budget=int(stem.split("_")[-1])
                except ValueError:
                    pass
            if nas_json.exists() and budget is not None:
                with open(nas_json) as f:
                    nr=json.load(f)
                info=nr.get(str(budget))
                if info:
                    arch=tuple(info["arch"])
            if arch is None:
                import random
                from src.pipeline.models import count_compressed_params
                anc_cfg=cfg.get("anchor", {})
                rng=random.Random(cfg["data"]["seed"])
                for _ in range(50000):
                    r1, r2, r3=(rng.choice(anc_cfg["rank_choices"]),)*3
                    f_, e_=rng.choice(anc_cfg["fusion_choices"]), rng.choice(anc_cfg["embed_choices"])
                    p_=count_compressed_params(r1, r2, r3, f_, e_, td, n_classes)
                    if abs(p_-anc_cfg["target_budget"]) / anc_cfg["target_budget"]<0.05:
                        arch=(r1, r2, r3, f_, e_)
                        break
            if arch is None:
                raise RuntimeError("Could not determine architecture of this checkpoint.") 
            teacher=TeacherCNN(input_shape=(1, 40, 40), num_classes=n_classes).to(device)
            teacher.load_state_dict(torch.load(root / cfg["teacher"]["weights_path"], map_location=device))
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad=False
            bridge=HSRBridge(teacher, input_shape=(1, 40, 40), tracker_dim=td, embed_dim=s1_cfg.get("embed_dim", 128), ranks=s1_cfg.get("hsr_ranks", [16, 8, 4]), dropout=s1_cfg.get("dropout", 0.4)).to(device)
            import torch.nn as nn
            bridge.classifier=nn.Linear(s1_cfg.get("embed_dim", 128), n_classes).to(device)
            bridge.load_state_dict(torch.load(root / s2_cfg.get("weights_path", "bridge_stage2.pth"), map_location=device))
            bridge.eval()
            for p in bridge.parameters():
                p.requires_grad=False
            r1, r2, r3, f_, e_=arch
            subnet=CompressedHSRBridge(bridge, r1, r2, r3, f_, e_, td, n_classes).to(device)
            subnet.load_state_dict(torch.load(self._path, map_location=device))
            subnet.eval()
            for p in subnet.parameters():
                p.requires_grad=False
            def infer(xf):
                return subnet(xf.to(device))
        
        B=cfg["data"].get("batch_size", 64)
        loader=DataLoader(TensorDataset(X_test, y_test), batch_size=B, shuffle=False)
        UNKNOWN_SENTINEL=splits["unknown_sentinel"]
        all_probs, all_preds, all_labels, all_energies=[], [], [], []
        temperature=1.0
        with torch.no_grad():
            for xf, lbl in loader:
                logits=infer(xf)
                probs=F.softmax(logits, dim=1)
                preds=logits.argmax(1)
                energy=-temperature*torch.logsumexp(logits/temperature, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbl.numpy())
                all_energies.extend(energy.cpu().numpy())
        labels_arr=np.array(all_labels)
        
        #Open-set
        val_loader=DataLoader(TensorDataset(X_val, y_val), batch_size=B, shuffle=False)
        val_energies, val_labels=[], []
        with torch.no_grad():
            for xf, lbl in val_loader:
                logits=infer(xf)
                energy=-temperature*torch.logsumexp(logits/temperature, dim=1)
                val_energies.extend(energy.cpu().numpy())
                val_labels.extend(lbl.numpy())
        ve=np.array(val_energies)
        vl=np.array(val_labels)
        unk_m=(vl==UNKNOWN_SENTINEL)
        kno_m=~unk_m
        known_energies=ve[kno_m]
        if len(known_energies)>0:
            best_T=float(np.percentile(known_energies, 95))
        else:
            best_T=-5.0
        
        with torch.no_grad():
            train_unk_energies = []
            for xb, in DataLoader(TensorDataset(splits["X_train_spatial_u"]), batch_size=256):
                logits = infer(xb)
                e = -temperature * torch.logsumexp(logits / temperature, dim=1)
                train_unk_energies.extend(e.cpu().numpy())
        train_unk_energies = np.array(train_unk_energies)
        print(f"[EVAL] train_unk_energies: mean={train_unk_energies.mean():.3f} median={np.median(train_unk_energies):.3f}")
        overlap_train = (train_unk_energies < best_T).mean()
        print(f"[EVAL] % di unknown DI TRAINING che sarebbero accettati come noti: {overlap_train*100:.1f}%")
        #Apply to test
        test_energies_arr=np.array(all_energies)
        test_probs_arr=np.array(all_probs)
        ki_arr=np.array(known_indices)
        final_preds=np.where(test_energies_arr>=best_T, UNKNOWN_SENTINEL, ki_arr[test_probs_arr[:, known_indices].argmax(1)])
        
        #Metrics
        cm=confusion_matrix(labels_arr, final_preds, labels=list(range(n_classes)))
        pre_cls=precision_score(labels_arr, final_preds, labels=list(range(n_classes)), average=None, zero_division=0)
        rec_cls=recall_score(labels_arr, final_preds, labels=list(range(n_classes)), average=None, zero_division=0)
        f1_cls=f1_score(labels_arr, final_preds, labels=list(range(n_classes)), average=None, zero_division=0)
        pcm={cls:{
            "precision": float(pre_cls[i]),
            "recall": float(rec_cls[i]),
            "f1": float(f1_cls[i])
        } for i, cls in enumerate(class_names)}
        macro_f1=float(f1_cls.mean())
        final_acc=float((final_preds==labels_arr).mean())
        kno_mask=labels_arr!=UNKNOWN_SENTINEL
        kn_acc=float((final_preds[kno_mask]==labels_arr[kno_mask]).mean()) if kno_mask.any() else 0.0
        unk_mask=labels_arr==UNKNOWN_SENTINEL
        unk_rec=float((final_preds[unk_mask]==UNKNOWN_SENTINEL).mean()) if unk_mask.any() else 0.0
        
        summary={
            "final_val_f1": macro_f1,
            "best_val_f1": macro_f1,
            "final_val_acc": final_acc,
            "final_train_f1": None,
            "final_train_acc": None,
            "final_val_loss": None,
            "final_train_loss": None,
            "threshold": best_T,
            "kn_acc": kn_acc,
            "unk_rec": unk_rec,
            "per_class": pcm,
        }
        sidecar_summary=self._summary
        summary.update({k: v for k, v in sidecar_summary.items() if k not in summary or summary[k] is None})
        return {"cm": cm, "pcm": pcm, "summary": summary}
                
    def _on_eval_done(self, result: Optional[dict], error: Optional[str]):
        self._eval_btn.configure(state="normal")
        self._spinner.configure(text="")
        if error:
            self._status.configure(text=f"{error}", text_color="#E24B4A")
            return
        self._cm=result["cm"]
        self._pcm=result["pcm"]
        self._summary.update(result["summary"])
        self._status.configure(text=f"Test macro-F1: {self._summary.get('final_val_f1', 0):.3f}  | Acc: {self._summary.get('final_val_acc', 0):.3f} | Threshold: {self._summary.get('threshold', 0):.2f}", text_color="#1D9E75")
        self._populate_tabs()
        
    def _populate_tabs(self):
        frame=self._tabs.tab("Confusion Matrix")
        for w in frame.winfo_children():
            w.destroy()
        ConfusionMatrixView(frame, self._cm, self._class_names).pack(fill="both", expand=True)
        
        frame=self._tabs.tab("Per-Class F1")
        for w in frame.winfo_children():
            w.destroy()
        PerClassF1View(frame, self._class_names, self._pcm).pack(fill="both", expand=True)
        
        frame=self._tabs.tab("Class Metrics")
        for w in frame.winfo_children():
            w.destroy()
        ClassMetricsTable(frame, self._class_names, self._pcm, fg_color="transparent").pack(fill="both", expand=True)
        
        frame=self._tabs.tab("Summary Log")
        for w in frame.winfo_children():
            w.destroy()
        SummaryLog(frame, self._summary, self._class_names).pack(fill="both", expand=True)

class ResultsPanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._selected_model: Optional[Path]=None
        self._model_btns: dict[str, ctk.CTkButton]={}
        self._build_content()
        app_state.subscribe("run_finished", self._on_run_finished)
        app_state.subscribe("stage_succeded", self._on_stage_succeeded)
        app_state.subscribe("dataset_prepared", self._on_dataset_changed)
        app_state.subscribe("dataset_changed", self._on_dataset_changed)
        self.after(50, self._on_dataset_changed)
        
    def _build_content(self):
        ctk.CTkLabel(self, text="Results", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(self, text="Benchmark results from the last evaluation run.", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(0, 12))
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 16))
        #Top-level
        self._top_tabs=ctk.CTkTabview(self, anchor="nw")
        self._top_tabs.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self._top_tabs.add("Benchmark")
        self._top_tabs.add("Evaluation")
        self._build_benchmark_tab(self._top_tabs.tab("Benchmark"))
        self._build_evaluation_tab(self._top_tabs.tab("Evaluation"))
        
    def _build_benchmark_tab(self, parent) -> None:
        #Action buttons
        btn_row=ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(8, 8))
        ctk.CTkButton(btn_row, text="Refresh", image=Icons.refresh, width=110, command=self._load_results).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Open benchmark plot", width=160, fg_color="transparent", border_width=1, command=self._open_plot).pack(side="left", padx=(0, 8))
        self._status_lbl=ctk.CTkLabel(btn_row, text="", font=ctk.CTkFont(size=11), text_color=("gray45", "gray55"), anchor="w")
        self._status_lbl.pack(side="left", padx=8)
        #Benchmark table
        tbl_frame=ctk.CTkScrollableFrame(self, fg_color=("gray94", "gray15"), corner_radius=10)
        tbl_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        cols=["Model", "Params", "Compr.", "Overall", "Known", "Unk F1", "Unk Rec", "Macro F1", "Threshold"]
        widths=[160, 90, 60, 80, 80, 80, 80, 80, 80]
        header=ctk.CTkFrame(tbl_frame, fg_color=("gray85", "gray25"))
        header.pack(fill="x", padx=4, pady=(4, 0))
        for c, w in zip(cols, widths):
            ctk.CTkLabel(header, text=c, width=w, anchor="center", font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=2)
        self._rows_frame=ctk.CTkFrame(tbl_frame, fg_color="transparent")
        self._rows_frame.pack(fill="both", expand=True, padx=4)
        self._col_widths=widths
        self._load_results()
        
    def _build_evaluation_tab(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)
        #Model List
        self._eval_gate_banner=ctk.CTkFrame(parent, fg_color="#3A2A1A", corner_radius=8)
        ctk.CTkLabel(self._eval_gate_banner, text="Dataset not prepared. Evaluation requires the fixed train/val/test split from the Data panel", font=ctk.CTkFont(size=12, weight="bold"), text_color="#EF9F27", anchor="w", wraplength=900).pack(fill="x", padx=14, pady=10)
        left=ctk.CTkFrame(parent, width=200, fg_color=("gray92", "gray14"), corner_radius=8)
        left.grid(row=1, column=0, sticky="nsew", padx=(4, 6), pady=4)
        left.grid_propagate(False)
        list_header=ctk.CTkFrame(left, fg_color="transparent")
        list_header.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(list_header, text="Models", font=ctk.CTkFont(size=12, weight="bold"), anchor="w").pack(side="left")
        ctk.CTkButton(list_header, text="", image=Icons.refresh, width=28, height=24, command=self._refresh_model_list).pack(side="right")
        self._model_list_frame=ctk.CTkScrollableFrame(left, fg_color="transparent", label_fg_color="transparent")
        self._model_list_frame.pack(fill="both", expand=True, padx=4, pady=(0, 8))
        #Metrics View
        self._right_frame=ctk.CTkFrame(parent, fg_color="transparent")
        self._right_frame.grid(row=1, column=1, sticky="nsew", pady=4)
        self._right_frame.rowconfigure(0, weight=1)
        self._right_frame.columnconfigure(0, weight=1)
        self._placeholder=ctk.CTkLabel(self._right_frame, text="Select a model from the list to evaluate it", font=ctk.CTkFont(size=13), text_color=("gray50", "gray55"))
        self._placeholder.grid(row=0, column=0)
        self._refresh_model_list()
        
    def _refresh_model_list(self) -> None:
        for w in self._model_list_frame.winfo_children():
            w.destroy()
        self._model_btns.clear()
        root=self.state.project_root / "models"
        paths=[]
        for fname in ["kws_multi_cnn_model_pytorch.pth", "refiner.pth", "bridge_stage1.pth", "bridge_stage2.pth", "anchor_1M.pth"]:
            p=root /fname
            if p.exists():
                paths.append(p)
        
        for p in sorted(root.glob("nas_bridge_*.pth")):
            paths.append(p)
        
        for p in sorted(root.glob("*.pth")):
            if p not in paths:
                paths.append(p)
                
        if not paths:
            ctk.CTkLabel(self._model_list_frame, text="No .pth found.\nRun pipeline first.", text_color=("gray50", "gray55"), font=ctk.CTkFont(size=11), justify="center").pack(pady=16)
            return
        
        for p in paths:
            stem=p.stem
            label=_PTH_META.get(stem, (stem, ""))[0] or stem
            short=label if len(label)<=20 else label[:18]+"..."
            btn=ctk.CTkButton(self._model_list_frame, text=short, anchor="w", fg_color="transparent", hover_color=("gray82", "gray22"), font=ctk.CTkFont(size=11), height=32, command=lambda path=p: self._select_model(path))
            btn.pack(fill="x", padx=4, pady=2)
            self._model_btns[str(p)]=btn
            
    def _select_model(self, path: Path) -> None:
        if not self.state.dataset_ready:
            for w in self._right_frame.winfo_children():
                w.destroy()
            ctk.CTkLabel(self._right_frame, text="Dataset not prepared. Go to the Data panel and click 'Prepare dataset' before running evaluation", font=ctk.CTkFont(size=13), text_color="#EF9F27", wraplength=500).grid(row=0, column=0)
            return
        
        for k, btn in self._model_btns.items():
            btn.configure(fg_color=("gray75", "gray28") if k==str(path) else "transparent")
        
        for w in self._right_frame.winfo_children():
            w.destroy()
            
        self._selected_model=path
        view=ModelMetricsView(self._right_frame, path, self.state)
        view.grid(row=0, column=0, sticky="nsew")
        self._right_frame.rowconfigure(0, weight=1)
        self._right_frame.columnconfigure(0, weight=1)
        
    def _on_dataset_changed(self, **_) -> None:
        ready=self.state.check_dataset_ready()
        if ready:
            self._eval_gate_banner.grid_forget()
        else:
            self._eval_gate_banner.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 0))
                
    def _load_results(self) -> None:
        path=self.state.project_root / "results" / "benchmark_results.json"
        for w in self._rows_frame.winfo_children():
            w.destroy()
        if not path.exists():
            ctk.CTkLabel(self._rows_frame, text="No benchmark_results.json found. Run Stage 6 first.", text_color=("gray50", "gray55"), font=ctk.CTkFont(size=12)).pack(pady=24)
            self._status_lbl.configure(text="No results file found.")
            return
        try:
            with open(path) as f:
                results=json.load(f)
        except Exception as e:
            self._status_lbl.configure(text=f"Load error: {e}", text_color="#E24B4A")
            return
        t_params=results[0]["params"] if results else 1
        odd=False
        for r in results:
            compr=f"{t_params/max(r['params'], 1):.1f}x"
            values=[
                r.get("label", "-"),
                f"{r['params']:,}",
                compr,
                f"{r['acc']*100:.1f}%",
                f"{r['kn_acc']*100:.1f}%",
                f"{r['unk_f1']*100:.1f}%",
                f"{r['unk_rec']*100:.1f}%",
                f"{r['macro_f1']*100:.1f}%",
                f"{r['threshold']:.2f}",
            ]
            row_bg=("gray90", "gray18") if odd else ("gray94", "gray15")
            row=ctk.CTkFrame(self._rows_frame, fg_color=row_bg)
            row.pack(fill="x", pady=1)
            for val, w in zip(values, self._col_widths):
                ctk.CTkLabel(row, text=val, width=w, anchor="center", font=ctk.CTkFont(size=11)).pack(side="left", padx=2)
            odd=not odd
        self._status_lbl.configure(text=f"Loaded {len(results)} models from {path.name}", text_color=("gray45", "gray55"))
        
    def _open_plot(self):
        cfg=self.state.config_data.get("eval", {})
        plot_path=self.state.project_root / "results" /cfg.get("benchmark_plot_path", "nas_benchmark.png")
        if not plot_path.exists():
            self._status_lbl.configure(text=f"Plot not found: {plot_path}", text_color="#E24B4A")
            return
        try:
            if sys.platform=="win32":
                subprocess.Popen(["start", str(plot_path)], shell=True)
            elif sys.platform=="darwin":
                subprocess.Popen(["open", str(plot_path)])
            else:
                subprocess.Popen(["xdg-open", str(plot_path)])
        except Exception as e:
            self._status_lbl.configure(text=f"Could not open plot: {e}", text_color="#E24B4A")

    def _on_run_finished(self, **_) -> None:
        self._load_results()
        self._refresh_model_list()
        
    def _on_stage_succeeded(self) -> None:
        self._refresh_model_list()