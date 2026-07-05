from __future__ import annotations
import json
import subprocess
import threading
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
from gui.panels.base_panel import BasePanel
from typing import Optional
import tkinter as tk
from gui.widgets.log_viewer import LogViewer
from src.assets import Icons
import sys

_SPLIT_COLORS={
    "train": "#378ADD",
    "val": "#1D9E75",
    "test": "#EF9F27"
}

SPEECH_COMMANDS_WORDS = sorted([
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", 
    "follow", "forward", "four", "go", "happy", "house", "learn", "left", 
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", 
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"
])

class ClassDistChart(ctk.CTkFrame):
    _BAR_W=10
    _GAP=4
    _GROUP=6
    _PAD_L=52
    _PAD_B=56
    _PAD_T=20
    _PAD_R=16
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=("gray94", "gray14"), corner_radius=10)
        self._canvas=tk.Canvas(self, bg="#1E1E1E", bd=0, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self._canvas.bind("<Configure>", lambda _: self._redraw())
        self._data: Optional[dict]=None
        
    def _redraw(self):
        c=self._canvas
        c.delete("all")
        if not self._data:
            c.create_text(int(c.winfo_width()/2) or 200, int(c.winfo_height()/2) or 100, text="Load a dataset to see class distribution", fill="#555", font=("Helvetica", 11))
            return
        W=c.winfo_width() or 600
        H=c.winfo_height() or 260
        classes=self._data["classes"]
        n=len(classes)
        splits=["train", "val", "test"]
        colors=[_SPLIT_COLORS[s] for s in splits]
        chart_w=W-self._PAD_L-self._PAD_R
        chart_h=H-self._PAD_T-self._PAD_B
        if chart_w<20 or chart_h<20:
            return
        #Auto-size
        group_w=chart_w/max(n, 1)
        bar_w=max(4, min(self._BAR_W, int(group_w/4)))
        gap=max(1, bar_w//3)
        trio_w=3*bar_w+2*gap
        max_val=max(max(self._data[s]) for s in splits if self._data[s]) or 1
        
        def fy(v):
            return self._PAD_T+chart_h-int(v/max_val*chart_h)
                    
        n_lines=5
        for i in range(n_lines+1):
            val=int(max_val*i/n_lines)
            y=fy(val)
            c.create_line(self._PAD_L, y, W-self._PAD_R, y, fill="#333", dash=(4, 6))
            c.create_text(self._PAD_L-4, y, text=str(val), anchor="e", fill="#777", font=("Helvetica", 8))
        #BARS
        for gi, cls in enumerate(classes):
            cx=self._PAD_L+(gi+0.5)*group_w
            x0=int(cx-trio_w/2)
            for bi, (split, color) in enumerate(zip(splits, colors)):
                val=self._data[split][gi]
                x1=x0+bi*(bar_w+gap)
                x2=x1+bar_w
                y1=fy(val)
                y2=fy(0)
                c.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                if(y2-y1)>12:
                    c.create_text((x1+x2)//2, y1-3, text=str(val), fill="#CCC", font=("Helvetica", 7), anchor="s")
            c.create_text(int(cx), H-self._PAD_B+6, text=cls, fill="#AAA", font=("Helvetica", 8), angle=35, anchor="ne")
            
        #Legend
        lx=self._PAD_L
        ly=H-12
        for split, color in zip(splits, colors):
            c.create_rectangle(lx, ly-8, lx+12, ly, fill=color, outline="")
            c.create_text(lx+15, ly-4, text=split, fill="#AAA", font=("Helvetica", 9), anchor="w")
            lx+=64
        #Axes
        c.create_line(self._PAD_L, self._PAD_T, self._PAD_L, H-self._PAD_B, fill="#555")
        c.create_line(self._PAD_L, H-self._PAD_B, W-self._PAD_R, H-self._PAD_B, fill="#555")
    
    def set_data(self, class_names: list[str], train_counts: list[int], val_counts: list[int], test_counts: list[int]) -> None:
        self._data={
            "classes": class_names,
            "train": train_counts,
            "val": val_counts,
            "test": test_counts
        }
        self._redraw()
        
    def clear(self) -> None:
        self._data=None
        self._canvas.delete("all")

class DataPanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._preparing=False
        self._checkboxes={}
        self._main_scroll=ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._main_scroll.pack(fill="both", expand=True)
        self._build_content()
        app_state.subscribe("config_loaded", self._on_config_loaded)
        self.after(50, self._check_existing)

    def _build_content(self) -> None:
        ctk.CTkLabel(self._main_scroll, text="Data", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(self._main_scroll, text="Configure dataset path and split", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(0, 16))
        sep=ctk.CTkFrame(self._main_scroll, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 16))
        #Top Half - CONFIG + STATS
        top=ctk.CTkFrame(self._main_scroll, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(0, 8))
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)
        cfg_col=ctk.CTkFrame(top, fg_color="transparent")
        cfg_col.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    
        opts_frame=ctk.CTkFrame(cfg_col, fg_color="transparent")
        opts_frame.pack(fill="x", padx=24, pady=(12, 0))
        ctk.CTkLabel(opts_frame, text="Feature Type:", width=90, anchor="w", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self._feat_menu=ctk.CTkOptionMenu(opts_frame, values=["MFE", "MFCC"], width=90)
        self._feat_menu.pack(side="left", padx=(0, 20))
        ctk.CTkLabel(opts_frame, text="Hardware: ", width=80, anchor="w", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self._hw_menu=ctk.CTkOptionMenu(opts_frame, values=["GPU (CUDA)", "CPU"], width=120)
        self._hw_menu.pack(side="left")
        #Class Grid
        ctk.CTkLabel(cfg_col, text="Target Words (Silence and Unknown auto-handled):", anchor="w", font=ctk.CTkFont(size=12, weight="bold")).pack(fill="x", padx=24, pady=(16, 4))
        self._scroll_grid=ctk.CTkScrollableFrame(cfg_col, height=140, fg_color=("gray85", "gray20"))
        self._scroll_grid.pack(fill="x", padx=24, pady=0)
        columns=4
        for i, word in enumerate(SPEECH_COMMANDS_WORDS):
            row=i//columns
            col=i%columns
            cb=ctk.CTkCheckBox(self._scroll_grid, text=word, width=80)
            cb.grid(row=row, column=col, padx=8, pady=6, sticky="w")
            self._checkboxes[word]=cb
        self._add_path_row(cfg_col, "Project root", "Root directory of the pipeline", "project_root")
        self._add_path_row(cfg_col, "Splits directory", "Folder containing pre-generated .npz splits", "dataset_path")
        #Project root 
        sep2=ctk.CTkFrame(cfg_col, height=1, fg_color=("gray80", "gray30"))
        sep2.pack(fill="x", padx=24, pady=(12, 12))
        ctk.CTkLabel(cfg_col, text="Split settings", font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(fill="x")
        self._split_rows: dict[str, ctk.CTkEntry]={}
        for k, l, d in [
            ("test_fraction", "Test fraction",       "Test fraction (e.g. 0.15)"),
            ("val_fraction",  "Val fraction",         "Val fraction (e.g. 0.15)"),
            ("seed",          "Random seed",          "Global RNG seed"),
            ("batch_size",    "Batch size",           "Mini-batch size for all stages"),
        ]:
            self._split_rows[k]=self._add_kv_row(cfg_col, l, d, k)
        #Status
        action_row=ctk.CTkFrame(cfg_col, fg_color="transparent")
        action_row.pack(fill="x", pady=(10, 0))
        self._status_lbl=ctk.CTkLabel(action_row, text="", anchor="w", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"))
        self._status_lbl.pack(fill="x", padx=(24, 10), pady=(16, 0))
        self._prepare_btn=ctk.CTkButton(action_row, text="Download & Extract", image=Icons.config, width=160, command=self._prepare)
        self._prepare_btn.pack(side="right", pady=(16, 0))
        #Stats column + Logger
        stats_col=ctk.CTkFrame(top, fg_color=("gray94", "gray16"), corner_radius=10)
        stats_col.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(stats_col, text="Dataset stats", font=ctk.CTkFont(size=12, weight="bold"), anchor="w").pack(fill="x", padx=12, pady=(10, 6))
        self._stat_labels: dict[str, ctk.CTkLabel]={}
        for key, caption in [
            ("total", "Total samples"),
            ("n_cls", "Classes"),
            ("tr_size", "Train samples"),
            ("va_size", "Val samples"),
            ("te_size", "Test samples"),
            ("feat", "Extracted Feat")
        ]:
            row=ctk.CTkFrame(stats_col, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=caption+":", font=ctk.CTkFont(size=11), text_color=("gray50", "gray55"), anchor="w", width=110).pack(side="left")
            lbl=ctk.CTkLabel(row, text="-", font=ctk.CTkFont(size=11, weight="bold"), anchor="w")
            lbl.pack(side="left")
            self._stat_labels[key]=lbl
        self._ready_badge=ctk.CTkLabel(stats_col, text="Not prepared", anchor="w", font=ctk.CTkFont(size=12, weight="bold"), text_color="#E24B4A")
        self._ready_badge.pack(fill="x", padx=12, pady=(8, 10))
        self._log_viewer=LogViewer(stats_col)
        self._log_viewer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        ctk.CTkFrame(self._main_scroll, height=1, fg_color=("gray80", "gray30")).pack(fill="x", padx=24, pady=(4, 8))
        ctk.CTkLabel(self._main_scroll, text="Class distribution (train/val/test)", font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(0, 4))
        #Chart
        self._chart=ClassDistChart(self._main_scroll)
        self._chart.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        self._refresh_from_state()
    
    def _get_active_manifest_path(self) -> Optional[Path]:
        cfg=self.state.config_data.get("data", {}) if self.state.config_data else {}
        ds_path=cfg.get("dataset_path", "data")
        p=Path(ds_path)
        if not p.is_absolute():
            p=self.state.project_root/p
        manifest_file=p/"manifest.json"
        if manifest_file.exists():
            return manifest_file
        return None
     
    def _prepare(self) ->None:
        if self._preparing:
            return
        selected_words=[w for w, cb in self._checkboxes.items() if cb.get()==1]
        print(selected_words)             
        if not selected_words:
            self._status_lbl.configure(text="Error: Please select at least one keyword!", text_color="#E24B4A")
            return
        if self.state.config_data and "data" in self.state.config_data:
            final_words=selected_words.copy()
            if "silence" not in final_words:
                final_words.append("silence")
            self.state.config_data["data"]["class_names"]=sorted(final_words)
            self.state.config_data["data"]["num_classes"]=len(final_words)
        cfg_path=self.state.abs_config()
        if not cfg_path.exists():
            self._status_lbl.configure(text=f"Config not found: {cfg_path}", text_color="#E24B4A")
            return
        try:
            import config.config as pcfg
            pcfg.save(self.state.config_data, cfg_path)
        except Exception as e:
            self._status_lbl.configure(text=f"Could not save config: {e}", text_color="#E24B4A")
            return
        feat_type=self._feat_menu.get()
        use_gpu="GPU" in self._hw_menu.get()
        self._preparing=True
        self._log_viewer.clear() 
        self._prepare_btn.configure(state="disabled", text="Extracting...", image=Icons.refresh)
        self._status_lbl.configure(text="Downloading and Extracting features...", text_color=("gray50", "gray55"))
        t=threading.Thread(target=self._prepare_worker, args=(cfg_path, feat_type, use_gpu), daemon=True)
        t.start()
        
    def _prepare_worker(self, cfg_path: Path, feat_type: str, use_gpu: bool) -> None:
        root=self.state.project_root
        script=root/"src"/"pipeline"/"prepare_dataset.py"
        if not script.exists():
            script=Path(__file__).resolve().parents[3]/"src"/"pipeline"/"prepare_dataset.py"
        cmd=[sys.executable, "-u", str(script), "--config", str(cfg_path), "--root", str(root), "--features", feat_type.lower()]
        if not use_gpu:
            cmd.append("--cpu-only")
        try:
            proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(root)) 
            buf=[]
            replace_line=False
            while True:
                char=proc.stdout.read(1)
                if not char:
                    break
                if char=='\r':
                    line=''.join(buf).strip()
                    if line:
                        self.after(0, self._log_viewer.append, line, replace_line)
                        replace_line=True
                    buf=[]
                elif char=='\n':
                    line=''.join(buf).strip()
                    if line:
                        self.after(0, self._log_viewer.append, line, replace_line)
                        replace_line=False
                        buf=[]
                else:
                    buf.append(char)
            proc.wait()
            self.after(0, self._on_prepare_done, proc.returncode)
        except Exception as e:
            self.after(0, self._log_viewer.append, f"ERROR: {str(e)}", False)
            self.after(0, self._on_prepare_done, -1)
            
    def _on_prepare_done(self, returncode: int) -> None:
        self._preparing=False
        self._prepare_btn.configure(state="normal", text="Download & Extract", image=Icons.config)
        if returncode!=0:
            self._status_lbl.configure(text=f"Preparation failed:", text_color="#E24B4A")
            self.state.dataset_ready=False
            self.state.emit("dataset_changed")
            return
        self._status_lbl.configure(text="Dataset prepared - dataset_train.npz / dataset_val.npz / dataset_test.npz written", text_color="#1D9E75")
        self.state.check_dataset_ready()
        self.state.emit("dataset_prepared")
        
    def _check_existing(self) -> None:
        manifest_path=self._get_active_manifest_path()
        if manifest_path:
            try:
                with open(manifest_path, 'r') as f:
                    m=json.load(f)
                self.state.dataset_manifest=m
                self.state.dataset_ready=True
                self._load_manifest_into_ui()
                self._status_lbl.configure(text=f"Loaded existing splits from {manifest_path.parent.name}/", text_color="#1D9E75")
            except Exception as e:
                self.state.dataset_ready=False
                self._status_lbl.configure(text=f"Error reading manifest: {e}", text_color="#E24B4A")
        else:
            self.state.dataset_ready=False
            self.state.dataset_manifest=None
            self._chart.clear()
            self._status_lbl.configure(text="No prepared splits found in target directory.", text_color="#EF9F27")
        self._update_ready_badge()
        self.state.emit("dataset_changed")

    def _load_manifest_into_ui(self) -> None:
        m=self.state.dataset_manifest
        if not m:
            return
        class_names=m.get("class_names", [])
        for word, cb in self._checkboxes.items():
            if word in class_names:
                cb.select()
            else:
                cb.deselect()
        if self.state.config_data and "data" in self.state.config_data:
            self.state.config_data["data"]["class_names"]=[c for c in class_names if c not in ("unknown",)]
            self.state.config_data["data"]["num_classes"]=m.get("num_classes", len(class_names))
            try:
                import config.config as pcfg
                pcfg.save(self.state.config_data, self.state.abs_config())
            except Exception:
                pass
        
        def counts_for(split_key):
            cc=m.get(f"{split_key}_class_counts", {})
            return [cc.get(c, 0) for c in class_names]
        
        self._chart.set_data(class_names, counts_for("train"), counts_for("val"), counts_for("test"))
        tr_unk=m.get("train_unknown_samples", 0)
        va_unk=m.get("val_unknown_samples", 0)
        te_unk=m.get("test_unknown_samples", 0)
        f_type=m.get("feature_type", "MFE")
        tr_tot=m.get("train_samples", 0)
        tr_known=tr_tot-tr_unk
        va_tot=m.get("val_samples", 0)
        va_known=va_tot-va_unk
        te_tot=m.get("test_samples", 0)
        te_known=te_tot-te_unk
        tot_known=tr_known+va_known+te_known
        tot_unk=tr_unk+va_unk+te_unk
        self._stat_labels["total"].configure(text=f"{tot_known:,} known + {tot_unk:,} unknown")
        self._stat_labels["n_cls"].configure(text=f"({m.get('num_classes', 0)} known + 1 unknown)")
        self._stat_labels["tr_size"].configure(text=f"{tr_known:,} known + {tr_unk} unk")
        self._stat_labels["va_size"].configure(text=f"{va_known:,} known + {va_unk} unk")
        self._stat_labels["te_size"].configure(text=f"{te_known:,} known + {te_unk} unk")
        self._stat_labels["feat"].configure(text=f_type, text_color="#378ADD")
        self._update_ready_badge()
        
    def _update_ready_badge(self) -> None:
        if self.state.dataset_ready:
            self._ready_badge.configure(text="Prepared & current", text_color="#1D9E75")
        else:
            self._ready_badge.configure(text="Not prepared", text_color="#E24B4A")
            
    def _add_kv_row(self, parent, label, desc, key) -> ctk.CTkEntry:
        frame=ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=24, pady=4)
        frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=label, width=140, anchor="w", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        entry=ctk.CTkEntry(frame, font=ctk.CTkFont(family="Courier New", size=12), width=120)
        entry.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ctk.CTkLabel(frame, text=desc, anchor="w", font=ctk.CTkFont(size=10), text_color=("gray50", "gray55")).grid(row=1, column=0, columnspan=2, sticky="w", pady=(1, 0))
        entry.bind("<FocusOut>", lambda e, k=key, w=entry: self._kv_changed(k, w.get()))
        return entry

    def _kv_changed(self, key: str, value: str) -> None:
        if self.state.config_data is None or "data" not in self.state.config_data:
            return
        try:
            orig=self.state.config_data["data"].get(key)
            from config.config import coerce_value
            self.state.config_data["data"][key]=coerce_value(value, orig)
        except Exception:
            pass
        
    def _add_path_row(self, parent, label: str, desc: str, key: str) -> None:
        frame=ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=24, pady=6)
        frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=label, width=140, anchor="w", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        entry=ctk.CTkEntry(frame, font=ctk.CTkFont(family="Courier New", size=11))
        entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        btn=ctk.CTkButton(frame, text="Browse", width=70, command=lambda k=key, e=entry: self._browse(k, e))
        btn.grid(row=0, column=2, sticky="e")
        ctk.CTkLabel(frame, text=desc, anchor="w", font=ctk.CTkFont(size=10), text_color=("gray50", "gray55"))
        entry.bind("<FocusOut>", lambda e, k=key, w=entry: self._path_changed(k, w.get()))
        if key=="project_root":
            self._root_entry=entry
        else: 
            self._npz_entry=entry
          
    def _path_changed(self, key: str, value: str) -> None:
        if key=="project_root":
            self.state.project_root=Path(value)
        else:
            if self.state.config_data is not None and "data" in self.state.config_data:        
                self.state.config_data["data"]["dataset_path"]=value
        self._check_existing()
        self.state.emit("data_changed")
    
    def _browse(self, key: str, entry: ctk.CTkEntry) -> None:
        from tkinter import filedialog
        if key=="project_root":
            path=filedialog.askdirectory(title="Select project root")
        else:
            path=filedialog.askdirectory(title="Select Splits Directory (containing dataset_*.npz)")
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)
            self._path_changed(key, path)
            
    def _on_config_loaded(self, **_) -> None:
        self._refresh_from_state()
        self._check_existing()
        
    def _refresh_from_state(self) -> None:
        config_source=self.state.config_data if self.state.config_data is not None else {}
        cfg=config_source.get("data", {})
        if hasattr(self, "_root_entry"):
            self._root_entry.delete(0, "end")
            self._root_entry.insert(0, str(self.state.project_root))
        if hasattr(self, "_npz_entry"):
            self._npz_entry.delete(0, "end")
            ds_path=cfg.get("dataset_path", "data")
            p=Path(ds_path)
            if not p.is_absolute():
                p=self.state.project_root/p
            self._npz_entry.insert(0, str(p))
            
        active_classes=cfg.get("class_names", [])
        for word, cb in self._checkboxes.items():
            if word in active_classes:
                cb.select()
            else:
                cb.deselect()
        for key, entry in self._split_rows.items():
            entry.delete(0, "end")
            entry.insert(0, str(cfg.get(key, "")))