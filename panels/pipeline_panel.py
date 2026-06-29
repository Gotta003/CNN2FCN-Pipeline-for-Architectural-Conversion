from __future__ import annotations
import customtkinter as ctk
import tkinter as tk
from gui.state import AppState
from config.runner import STAGE_DEFS, STAGE_DEPS
from panels.base_panel import BasePanel

_STAGE_DESCRIPTIONS = {
    "teacher": "Train the Teacher CNN on the spectrogram dataset.",
    "enforce": "Pre-train the ENFORCE LogitRefiner on teacher logits.\n"
               "Produces normalised latent targets used in hint distillation.",
    "hint":    "Stage 2 — HSR hint distillation.\n"
               "Train the Bridge DNN using refined latent targets + focal loss.",
    "dkd":     "Stage 3 — DKD fine-tuning.\n"
               "Fine-tune Bridge relaxation layers with TCKD + NCKD losses.",
    "anchor":  "Stage 4 — 1M anchor student.\n"
               "Train a CompressedHSRBridge at 1M params; provides warm-start weights for NAS.",
    "nas":     "Stage 5 — Evolutionary NAS.\n"
               "Tournament selection + crossover + mutation over architecture budgets.\n"
               "Proxy-trains candidates, fully fine-tunes each budget winner.",
    "eval":    "Stage 6 — Evaluation.\n"
               "Tune open-set confidence thresholds, run benchmark, save JSON + plot.",
}

class PipelinePanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._switches: dict[str, ctk.CTkSwitch]={}
        self._vars: dict[str, ctk.BooleanVar]={}
        self._build_content()
        
    def _build_content(self):
        ctk.CTkLabel(self, text="Pipeline stages", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(self, text="Toggle which stages to run. Disabling a stage also disables its dependents", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(0, 16))
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 16))
        #Quick actions
        btn_row=ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(0, 16))
        ctk.CTkButton(btn_row, text="Enable all", width=110, command=self._enable_all).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Disable all", width=110, fg_color="transparent", border_width=1, command=self._disable_all).pack(side="left")
        
        #Stage cards
        for stage_id, display_name, _ in STAGE_DEFS:
            var=tk.BooleanVar(value=self.state.stage_enabled.get(stage_id, True))
            self._vars[stage_id]=var
            self._build_card(stage_id, display_name, var)
            
    def _build_card(self, stage_id: str, display_name: str, var) -> None:
        card=ctk.CTkFrame(self, corner_radius=10, fg_color=("gray94", "gray17"))
        card.pack(fill="x", padx=24, pady=5)
        top=ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(10, 4))
        top.columnconfigure(0, weight=1)
        sw=ctk.CTkSwitch(top, text=display_name, variable=var, font=ctk.CTkFont(size=13, weight="bold"), command=lambda sid=stage_id: self._on_toggle(sid))
        sw.grid(row=0, column=0, sticky="w")
        self._switches[stage_id]=sw
        #Dependencies
        deps=STAGE_DEPS.get(stage_id, [])
        if deps:
            dep_txt="Requires: " + ", ".join(deps)
            ctk.CTkLabel(top, text=dep_txt, anchor="e", font=ctk.CTkFont(size=10), text_color=("gray50", "gray55")).grid(row=0, column=1, sticky="e")
        desc=_STAGE_DESCRIPTIONS.get(stage_id, "")
        if desc:
            ctk.CTkLabel(card, text=desc, anchor="w", justify="left", font=ctk.CTkFont(size=11), text_color=("gray45", "gray60"), wraplength=620).pack(fill="x", padx=16, pady=(0, 10))  
        
    def _on_toggle(self, id: str) ->None:
        enabled=self._vars[id].get()
        self.state.stage_enabled[id]=enabled
        if not enabled:
            for sid, _, _ in STAGE_DEFS:
                if id in STAGE_DEPS.get(sid, []):
                    self._vars[sid].set(False)
                    self.state.stage_enabled[sid]=False
        else:
            for dep in STAGE_DEPS.get(id, []):
                if not self.state.stage_enabled.get(dep, False):
                    self._vars[id].set(False)
                    self.state.stage_enabled[id]=False
                    return
        self.state.emit("stages_changed")
        
    def _enable_all(self):
        for sid in self._vars:
            self._vars[sid].set(True)
            self.state.stage_enabled[sid]=True
        self.state.emit("stages_changed")
    
    def _disable_all(self):
        for sid in self._vars:
            self._vars[sid].set(False)
            self.state.stage_enabled[sid]=False
        self.state.emit("stages_changed")