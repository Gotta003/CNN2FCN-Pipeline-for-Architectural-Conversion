from __future__ import annotations
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
from panels.base_panel import BasePanel
from typing import Optional
import tkinter as tk
from src.assets import Icons

_SPLIT_COLORS={
    "train": "#378ADD",
    "val": "#1D9E75",
    "test": "#EF9F27"
}

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
        c.create_line(self._PAD_L, self._PAD_T, self._PAD_L, H-self._PAD_B, fill="#555", width=1)
        c.create_line(self._PAD_L, H-self._PAD_B, W-self._PAD_R, H-self._PAD_B, fill="#555", width=1)
    

class DataPanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._build_content()
        app_state.subscribe("config_loaded", self._on_config_loaded)

    def _build_content(self) -> None:
        pad={"padx": 24, "pady": 6}
        ctk.CTkLabel(self, text="Data", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(self, text="Configure dataset path and split", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(0, 16))
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 16))
        #Top Half - CONFIG + STATS
        top=ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(0, 8))
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=1)
        cfg_col=ctk.CTkFrame(top, fg_color="transparent")
        cfg_col.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        #Project root 
        self._add_path_row(cfg_col, "Project root", "Root directory of te pipeline", "project_root")
        self._add_path_row(cfg_col, "Dataset (.npz)", "Path to the KWS spectrogram dataset", "dataset_path")
        sep2=ctk.CTkFrame(cfg_col, height=1, fg_color=("gray80", "gray30"))
        sep2.pack(fill="x", padx=24, pady=(12, 12))
        ctk.CTkLabel(cfg_col, text="Split settings", font=ctk.CTkFont(size=13, weight="bold"), anchor="w").pack(fill="x", **pad)
        self._split_rows: dict[str, ctk.CTkEntry]={}
        for k, l, d in [
            ("test_fraction", "Test fraction",       "Held-out test fraction (e.g. 0.30)"),
            ("val_fraction",  "Val fraction",         "Val fraction of the remaining split"),
            ("seed",          "Random seed",          "Global RNG seed"),
            ("batch_size",    "Batch size",           "Mini-batch size for all stages"),
        ]:
            self._split_rows[k]=self._add_kv_row(cfg_col, l, d, k)
        #Status
        action_row=ctk.CTkFrame(cfg_col, fg_color="transparent")
        action_row.pack(fill="x", pady=(10, 0))
        self._status_lbl=ctk.CTkLabel(action_row, text="", anchor="w", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"))
        self._status_lbl.pack(fill="x", padx=24, pady=(16, 0))
        ctk.CTkButton(action_row, text="Preview dataset", image=Icons.refresh, width=150, command=self._preview).pack(side="right")

        
        
        self._refresh_from_state()
        
    def _preview(self):
        pass
            
    def _add_kv_row(self, label, desc, key) -> ctk.CTkEntry:
        frame=ctk.CTkFrame(self, fg_color="transparent")
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
        
    def _add_path_row(self, label: str, desc: str, key: str) -> None:
        frame=ctk.CTkFrame(self, fg_color="transparent")
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
        self._validate()
        self.state.emit("data_changed")
    
    def _browse(self, key: str, entry: ctk.CTkEntry) -> None:
        from tkinter import filedialog
        if key=="project_root":
            path=filedialog.askdirectory(title="Select project root")
        else:
            path=filedialog.askopenfilename(
                title="Select dataset .npz", filetypes=[("NumPy compressed", "*.npz"), ("All", "*")]
            )
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)
            self._path_changed(key, path)
            
    def _on_config_loaded(self, **_) -> None:
        self._refresh_from_state()
        
    def _refresh_from_state(self) -> None:
        config_source=self.state.config_data if self.state.config_data is not None else {}
        cfg=config_source.get("data", {})
        if hasattr(self, "_root_entry"):
            self._root_entry.delete(0, "end")
            self._root_entry.insert(0, str(self.state.project_root))
        if hasattr(self, "_npz_entry"):
            self._npz_entry.delete(0, "end")
            path=str(self.state.project_root) + cfg.get("dataset_path", "")
            self._npz_entry.insert(0, path)
        for key, entry in self._split_rows.items():
            entry.delete(0, "end")
            entry.insert(0, str(cfg.get(key, "")))
        self._validate()
        
    def _validate(self):
        config_source=self.state.config_data if self.state.config_data is not None else {}
        cfg=config_source.get("data", {})
        root=self.state.project_root
        if not root:
            self._status_lbl.configure(text="Project root not set", text_color="#E24B4A")
            return
        dataset_val = cfg.get("dataset_path", "")
        if not dataset_val:
            self._status_lbl.configure(text="Dataset path placeholder empty", text_color=("gray40", "gray60"))
            return
        npz=Path(cfg.get("dataset_path", ""))
        if not npz.is_absolute():
            npz=root/npz
        if npz.exists():
            self._status_lbl.configure(text=f"Dataset found: {npz.name}", text_color="#1D9E75")
        else:
            self._status_lbl.configure(text=f"Dataset not found: {npz}", text_color="#E24B4A")
            