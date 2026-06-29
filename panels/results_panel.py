from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
from panels.base_panel import BasePanel
from src.assets import Icons
import config.config as pcfg

class ResultPanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._build_content()
        app_state.subscribe("run_finished", self._on_run_finished)
        
    def _build_content(self):
        ctk.CTkLabel(self, text="Results", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(self, text="Benchmark results from the last evaluation run.", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(0, 12))
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 16))
        #Action buttons
        btn_row=ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(0, 16))
        ctk.CTkButton(btn_row, text="Refresh", image=Icons.refresh, width=110, command=self._load_results).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Open benchmark plot", width=160, fg_color="transparent", border_width=1, command=self._open_plot).pack(side="left", padx=(0, 8))
        self._status_lbl=ctk.CTkLabel(btn_row, text="", font=ctk.CTkFont(size=11), text_color=("gray45", "gray55"), anchor="w")
        self._status_lbl.pack(side="left", padx=8)
        #Benchmark table
        tbl_frame=ctk.CTkScrollableFrame(self, fg_color=("gray94", "gray15"), corner_radius=10)
        tbl_frame.pack(fill="both", expand=True, padx=24, pady=(0, 16))
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