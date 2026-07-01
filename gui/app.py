"""
Root window with sidebar navigation that manipulates the main windows
"""

from __future__ import annotations
import importlib
import threading
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
from src.assets import Icons

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

Icons.load_all()

NAV_ITEMS=[
    ("data", "Data", Icons.dataset),
    ("config", "Config", Icons.config),
    ("pipeline", "Pipeline", Icons.pipeline),
    ("run", "Run", Icons.run),
    ("results", "Results", Icons.results),
    ("netron", "Netron", Icons.hexagon),
]

_PANEL_REGISTRY: dict[str, tuple[str, str]]={
    "data": ("gui.panels.data_panel", "DataPanel"),
    "config": ("gui.panels.config_panel", "ConfigPanel"),
    "pipeline": ("gui.panels.pipeline_panel", "PipelinePanel"),
    "run": ("gui.panels.run_panel", "RunPanel"),
    "results": ("gui.panels.results_panel", "ResultsPanel"),
    "netron": ("gui.panels.netron_panel", "NetronPanel"),
}

class LoadingPlaceholder(ctk.CTkFrame):
    def __init__(self, master, label: str, **kw):
        super().__init__(master, **kw)
        self.configure(fg_color="transparent")
        ctk.CTkLabel(self, text=f"Loading {label}...", font=ctk.CTkFont(size=13), text_color=("gray50", "gray55")).place(relx=0.5, rely=0.5, anchor="center")

class App(ctk.CTk):
    def __init__(self, project_root: Path):
        super().__init__()
        self.title("CNN2FCN - ENFORCE + EvoNAS Pipeline")
        self.geometry("1200x780")
        self.minsize(900, 620)
        self.state=AppState(project_root=project_root)
        self._panels: dict[str, ctk.CTkFrame | None]={k: None for k  in _PANEL_REGISTRY}
        self._placeholders: dict[str, LoadingPlaceholder]={}
        self._active: str="data"
        self._buttons: dict[str, ctk.CTkButton]={}
        self._build_layout()
        self.after(0, self._deferred_init)
        
    def _deferred_init(self) -> None:
        self._load_config()
        self.state.check_dataset_ready()
        self._switch("data")
        self.state.subscribe("open_netron", self._on_open_netron)    
        
    def _build_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        #Sidebar Location
        sidebar=ctk.CTkFrame(self, width=180, corner_radius=0, fg_color=("gray90", "gray12"))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        #Logo/Title
        logo_frame=ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=12, pady=(18, 20))
        ctk.CTkLabel(logo_frame, text="ENFORCE", font=ctk.CTkFont(size=15, weight="bold"), anchor="w").pack(fill="x")
        ctk.CTkLabel(logo_frame, text="+", font=ctk.CTkFont(size=15, weight="bold"), anchor="w").pack(fill="x")
        ctk.CTkLabel(logo_frame, text="EvoNas Pipeline", font=ctk.CTkFont(size=15), text_color=("gray50", "gray55"), anchor="w").pack(fill="x")
        #Button NAV
        for nav_id, label, icon in NAV_ITEMS:
            if nav_id=="netron":
                ctk.CTkFrame(sidebar, height=1, fg_color=("gray80", "gray25")).pack(fill="x", padx=8, pady=(6,0))
            btn=ctk.CTkButton(
                sidebar,
                text=f"{label}",
                image=icon,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray82", "gray22"),
                text_color=("gray20", "gray80"),
                font=ctk.CTkFont(size=13),
                height=38,
                corner_radius=6,
                command=lambda nid=nav_id: self._switch(nid),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._buttons[nav_id]=btn
        #Show .pth ready
        self._netron_badge=ctk.CTkLabel(sidebar, text="", font=ctk.CTkFont(size=10), text_color="#1D9E75", anchor="w")
        self._netron_badge.pack(fill="x", padx=20, pady=(0,4))
        #Bottom: toggling options
        sidebar.pack_propagate(False)
        bottom=ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=8, pady=12)
        ctk.CTkLabel(bottom, text="Appearance", font=ctk.CTkFont(size=10), text_color=("gray50", "gray55")).pack(anchor="w", padx=4)
        ctk.CTkOptionMenu(bottom, values=["Dark", "Light", "System"], command=lambda v: ctk.set_appearance_mode(v.lower()), height=28).pack(fill="x", pady=(2,0))
        
        # MAIN AREA
        self._content=ctk.CTkFrame(self, fg_color=("gray96", "gray10"), corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew")
        self._content.grid_rowconfigure(0, weight=1)
        self._content.grid_columnconfigure(0, weight=1)
    
    def _get_panel(self, nav_id: str):
        if self._panels[nav_id] is not None:
            return self._panels[nav_id]
        if nav_id not in self._placeholders:
            label=next(l for nid, l, _ in NAV_ITEMS if nid==nav_id)
            ph=LoadingPlaceholder(self._content, label)
            ph.grid(row=0, column=0, sticky="nsew")
            self._placeholders[nav_id]=ph
            
        def _load():
            mod_name, cls_name=_PANEL_REGISTRY[nav_id]
            mod=importlib.import_module(mod_name)
            cls=getattr(mod, cls_name)
            self.after(0, self._install_panel, nav_id, cls)
        
        t=threading.Thread(target=_load, daemon=True)
        t.start()
        return self._placeholders[nav_id]
    
    def _install_panel(self, nav_id: str, cls) -> None:
        panel=cls(self._content, app_state=self.state)
        panel.grid(row=0, column=0, sticky="nsew")
        self._panels[nav_id]=panel
        if self._active==nav_id:
            panel.tkraise()
        ph=self._placeholders.pop(nav_id, None)
        if ph:
            ph.destroy()
        
    def _update_netron_badge(self, stage_id: str="", **_) -> None:
        n=sum(1 for v in self.state.netron_ready.values() if v)
        self._netron_badge.configure(text=f"    {n} checkpoint{'s' if n!=1 else ''} ready" if n else "")    
    
    def _switch(self, id: str) -> None:
        self._active=id
        frame=self._get_panel(id)
        frame.tkraise()
        active_fg=("gray75", "gray28")
        inactive_fg="transparent"
        for nid, btn in self._buttons.items():
            is_active=nid==id
            btn.configure(fg_color=active_fg if is_active else inactive_fg, font=ctk.CTkFont(size=13, weight="bold" if is_active else "normal"))       
    
    def _load_config(self) -> None:
        import config.config as pcfg
        cfg_path=self.state.project_root/"config"/"pipeline.yaml"
        if cfg_path.exists():
            try:
                self.state.config_data=pcfg.load(cfg_path)
                self.state.config_path=Path("config/pipeline.yaml")
                self.state.emit("config_loader")
            except Exception as e:
                print(f"[WARN] Could not load config: {e}")
        else:
            print(f"[INFO] Config not found at {cfg_path}. Set project root and reload in config panel.")
        self.state.subscribe("stage_succeeded", self._update_netron_badge)
        
    def _on_open_netron(self, stage_id: str="", **_) -> None:
        self._switch("netron")