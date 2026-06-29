from __future__ import annotations
import queue
import re
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
from gui.widgets.log_viewer import LogViewer
from gui.widgets.loss_canvas import LossCanvas
from config.runner import StageRunner, StageStatus, STAGE_DEFS
from src.assets import Icons
from panels.base_panel import BasePanel

# Training lines are:
# Ep 12 | loss=0.4321 | train=87.34% val=82.10%
_EP_RE=re.compile(r"Ep\s+(\d+).*?loss=([0-9.]+).*?val=([0-9.]+)%", re.IGNORECASE)

#Stages with PTH
_HAS_PTH={"teacher", "enforce", "hint", "dkd", "anchor", "nas"}

_SHORT={
    "teacher": "Teacher",
    "enforce": "ENFORCE",
    "hint": "Hint KD",
    "dkd": "DKD",
    "anchor": "Anchor",
    "nas": "NAS",
    "eval": "Eval",
}

_STATUS_COLORS = {
    StageStatus.IDLE:    ("gray75", "gray35"),
    StageStatus.RUNNING: ("#EF9F27", "#EF9F27"),
    StageStatus.SUCCESS: ("#1D9E75", "#1D9E75"),
    StageStatus.FAILED:  ("#E24B4A", "#E24B4A"),
    StageStatus.SKIPPED: ("gray60",  "gray45"),
}

class RunPanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._runner: StageRunner | None=None
        self._log_queue: queue.Queue=queue.Queue()
        self._active_stage: str | None=None
        self._stage_cells: dict[str, dict]={}
        self._status_icons={
            StageStatus.IDLE: Icons.circle,
            StageStatus.RUNNING: Icons.refresh,
            StageStatus.SUCCESS: Icons.check,
            StageStatus.FAILED: Icons.error,
            StageStatus.SKIPPED: Icons.skip,
        }
        self._build_content()
        
    def _build_content(self):
        header=ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 4))
        header.columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="Run", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").grid(row=0, column=0, sticky="w")
        btn_frame=ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=2, sticky="e")
        self._run_btn=ctk.CTkButton(btn_frame, text="Run pipeline", image=Icons.run, width=140, command=self._start)
        self._abort_btn=ctk.CTkButton(btn_frame, text="Abort", image=Icons.stop, width=90, fg_color="#C0392B", hover_color="#922B21", state="disabled", command=self._abort)
        self._abort_btn.pack(side="left")
        ctk.CTkLabel(self, text="Runs the enabled stages sequentially. Each stage is a separate subprocess.", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w", wraplength=900).pack(fill="x", padx=24, pady=(0, 12))
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 12))
        #Process Strip
        self._strip=ctk.CTkScrollableFrame(self, height=78, orientation="horizontal", fg_color=("gray92", "gray15"))
        self._strip.pack(fill="x", padx=24, pady=(0, 12))
        for sid, _, _ in STAGE_DEFS:
            self._build_stage_cell(sid) 
        #Log left, chart right
        split=ctk.CTkFrame(self, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        split.columnconfigure(0, weight=3)
        split.columnconfigure(1, weight=2)
        split.rowconfigure(0, weight=1)
        log_frame=ctk.CTkFrame(split, fg_color="transparent")
        log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ctk.CTkLabel(log_frame, text="Live output", font=ctk.CTkFont(size=11), text_color=("gray45", "gray60"), anchor="w").pack(fill="x")
        self._log=LogViewer(log_frame, corner_radius=8)
        self._log.pack(fill="both", expand=True)
        
        chart_frame=ctk.CTkFrame(split, fg_color="transparent")
        chart_frame.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(chart_frame, text="Training metrics", font=ctk.CTkFont(size=11), text_color=("gray45", "gray60"), anchor="w").pack(fill="x")
        self._chart=LossCanvas(chart_frame, corner_radius=8)
        self._chart.pack(fill="both", expand=True)
        
    def _build_stage_cell(self, stage_id: str) -> None:
        cell=ctk.CTkFrame(self._strip, fg_color=("gray86", "gray20"), corner_radius=8)
        cell.pack(side="left", padx=5, pady=6)
        lbl=ctk.CTkLabel(cell, text=f"{self._status_icons[StageStatus.IDLE]}  {_SHORT[stage_id]}", font=ctk.CTkFont(size=11), text_color=_STATUS_COLORS[StageStatus.IDLE], width=90, anchor="center")
        lbl.pack(padx=8, pady=(6, 2))
        netron_btn=None
        if stage_id in _HAS_PTH:
            netron_btn=ctk.CTkButton(cell, text="Netron", image=Icons.hexagon, width=82, height=22, font=ctk.CTkFont(size=10), fg_color="transparent", border_width=1, border_color=("gray60", "gray45"), text_color=("gray50", "gray50"), state="disabled", command=lambda sid=stage_id: self._open_netron(sid))
            netron_btn.pack(padx=8, pady=(0, 6))
        self._stage_cells[stage_id]={"label": lbl, "netron_btn": netron_btn}
        
    def _start(self):
        if self.state.is_running:
            return
        cfg_path=self.state.abs_config()
        if not cfg_path.exists():
            self._log.append(f"[ERROR] Config not found: {cfg_path}")
            return
        enabled=[sid for sid, en in self.state.stage_enabled.items() if en]
        if not enabled:
            self._log.append("[WARN] No stages enabled")
            return
        self._log.clear()
        self._chart.clear()
        self._reset_stage_indicators()
        self.state.is_running=True
        self._run_btn.configure(state="disabled")
        self._abort_btn.configure(state="normal")
        self._runner=StageRunner(
            project_root=self.state.project_root,
            config_path=cfg_path,
            log_queue=self._log_queue,
            status_cb=self._on_stage_status,
        )
        self._runner.run(enabled_stages=enabled)
        self._poll_log()
        
    def _abort(self):
        if self._runner:
            self._runner.abort()
        self._finish()
    
    def _finish(self):
        self.state.is_running=False
        self._run_btn.configure(state="normal")
        self._abort_btn.configure(state="disabled")
        
    def _reset_stage_indicators(self):
        for id, cell in self._stage_cells.items():
            cell["label"].configure(
                text=f"{self._status_icons[StageStatus.IDLE]}  {_SHORT[id]}",
                text_color=_STATUS_COLORS[StageStatus.IDLE],
            )
            btn=cell.get("netron_btn")
            if btn:
                btn.configure(state="disabled", fg_color="transparent", border_color=("gray60", "gray45"), text_color=("gray50", "gray50"))
    
    def _open_netron(self, stage_id: str) -> None:
        self.state.emit("open_netron", stage_id=stage_id)
    
    def _poll_log(self):
        try:
            while True:
                line=self._log_queue.get_nowait()
                self._log.append(line)
                self._parse_metrics(line)
        except queue.Empty:
            pass
        
        if self._runner and self._runner.is_alive():
            self.after(80, self._poll_log)
        else:
            try:
                while True:
                    line=self._log_queue.get_nowait()
                    self._log.append(line)
                    self._parse_metrics(line)
            except queue.Empty:
                pass
            self._finish()
            self.state.emit("run_finished")
        
    def _parse_metrics(self, line) -> None:
        m=_EP_RE.search(line)
        if m and self._active_stage:
            epoch=int(m.group(1))
            loss=float(m.group(2))
            val_acc=float(m.group(3))/100.0
            self._chart.add_point(self._active_stage, epoch, loss, val_acc)
            
    def _on_stage_status(self, stage_id: str, status: StageStatus)->None:
        self.after(0, self._update_indicator, stage_id, status)
        if status==StageStatus.RUNNING:
            self._active_stage=stage_id
        elif status in (StageStatus.SUCCESS, StageStatus.FAILED, StageStatus.SKIPPED):
            if self._active_stage==stage_id:
                self._active_stage=None
        
    def _update_indicator(self, stage_id: str, status: StageStatus) -> None:
        cell=self._stage_cells.get(stage_id)
        if not cell:
            return
        lbl=cell["label"]
        lbl.configure(
            text=f"{self._status_icons[status]}  {_SHORT[stage_id]}",
            text_color=_STATUS_COLORS[status],
        )
        if status==StageStatus.SUCCESS and stage_id in _HAS_PTH:
            btn=cell.get("netron_btn")
            if btn:
                btn.configure(state="normal", fg_color=("gray20", "#1A3A2A"), border_color="#1D9E75", text_color="#1D9E75")
            self.state.netron_ready[stage_id]=True
            self.state.emit("stage_succedeed", stage_id=stage_id)