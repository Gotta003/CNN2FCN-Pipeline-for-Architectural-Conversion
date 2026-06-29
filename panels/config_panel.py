from __future__ import annotations
import customtkinter as ctk
from gui.state import AppState
from gui.widgets.yaml_form import YamlForm
import config.config as pcfg
from panels.base_panel import BasePanel

class ConfigPanel(BasePanel):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._form: YamlForm | None=None
        self._unsaved=False
        self._build_content()
        app_state.subscribe("config_loaded", self._on_config_loaded)
        
    def _build_content(self):
        top=ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 0))
        top.columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Configuration", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").grid(row=0, column=0, sticky="w")
        btn_frame=ctk.CTkFrame(top, fg_color="transparent")
        btn_frame.grid(row=0, column=2, sticky="e")
        self._save_btn=ctk.CTkButton(btn_frame, text="Save", width=80, command=self._save)
        self._save_btn.pack(side="right", padx=(8,0))
        self._reload_btn=ctk.CTkButton(btn_frame, text="Reload from disk", width=130, fg_color="transparent", border_width=1, command=self._reload)
        self._reload_btn.pack(side="right")
        ctk.CTkLabel(self, text="Edit pipeline hyperparameters. Changes applied on Save", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(4, 12))
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 0))
        self._status_lbl=ctk.CTkLabel(self, text="", anchor="w", font=ctk.CTkFont(size=11), text_color=("gray50", "gray55"))
        self._form_container=ctk.CTkFrame(self, fg_color="transparent")
        self._form_container.pack(fill="both", expand=True)
        self._rebuild_form()
    
    def _rebuild_form(self) -> None:
        for w in self._form_container.winfo_children():
            w.destroy()
        visible=[s for s in pcfg.SECTIONS if s[0] in self.state.config_data]
        if not visible:
            ctk.CTkLabel(self._form_container, text="No config loaded. Set project root and reload", text_color=("gray50", "gray55"), font=ctk.CTkFont(size=13)).pack(pady=60)
            return
        self._form=YamlForm(self._form_container, sections=visible, config_data=self.state.config_data, on_change=self._on_field_change, fg_color="transparent", label_fg_color="transparent")
        self._form.pack(fill="both", expand=True, padx=8, pady=4)
        
    def _on_field_change(self, section: str, key: str, raw: str):
        try:
            orig=self.state.config_data[section][key]
            from config.config import coerce_value
            self.state.config_data[section][key]=coerce_value(raw, orig)
            self._mask_unsaved()
        except (KeyError, Exception):
            pass
        
    def _mask_unsaved(self):
        if not self._unsaved:
            self._unsaved=True
            self._status_lbl.configure(text="Unsaved changes", text_color="#EF9F27")
    
    def _on_config_loaded(self):
        self._rebuild_form()
        self._status_lbl.configure(text=f"Loaded: {self.state.abs_config()}", text_color=("gray50", "gray55"))
        self._unsaved=False
    
    def _save(self):
        path=self.state.abs_config()
        try:
            pcfg.save(self.state.config_data, path)
            self._unsaved=False
            self._status_lbl.configure(text=f"Saved to {path}", text_color="#1D9E75")
        except Exception as e:
            self._status_lbl.configure(text=f"Save failed: {e}", text_color="#E24B4A")
    
    def _reload(self):
        path=self.state.abs_config()
        try:
            self.state.config_data=pcfg.load(path)
            self.state.emit("config_loaded")
            self._status_lbl.configure(text=f"Reloaded from {path}", text_color="#1D9E75")
        except Exception as e:
            self._status_lbl.configure(text="Reload failed: {e}", text_color="#E24B4A")