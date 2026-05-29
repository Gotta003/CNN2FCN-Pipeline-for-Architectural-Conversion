"""
Root window with sidebar navigation that manipulates the main windows
"""

from __future__ import annotations
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
import src.pipeline.config as pc
from panels.temp_panel import TempPanel
from panels.netron_panel import NetronPanel

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

NAV_ITEMS=[
    ("data", "Data", "*"),
    ("config", "Config", "*"),
    ("pipeline", "Pipeline", "*"),
    ("run", "Run", "*"),
    ("results", "Results", "*"),
    ("netron", "Netron", "*"),
]

class App(ctk.CTk):
    def __init__(self, project_root: Path):
        super().__init__()
        self.title("CNN2FCN - ENFORCE + EvoNAS Pipeline")
        self.geometry("1200x780")
        self.minsize(900, 620)
        self.state=AppState(project_root=project_root)
        self._load_config()
        self._panels: dict[str, ctk.CTkFrame]={}
        self._buttons: dict[str, ctk.CTkButton]={}
        self._active: str="data"
        self._build_layout()
        
        #Control
        print(self.state)
    
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
        #Separator
        sep=ctk.CTkFrame(sidebar, height=1, fg_color=("gray80", "gray25"))
        sep.pack(fill="x", padx=8, pady=(0,12))
        #Button NAV
        for nav_id, label, icon in NAV_ITEMS:
            extra={}
            if nav_id=="netron":
                ctk.CTkFrame(sidebar, height=1, fg_color=("gray80", "gray25")).pack(fill="x", padx=8, pady=(6,0))
            btn=ctk.CTkButton(
                sidebar,
                text=f" {icon}  {label}",
                anchor="w",
                fg_color="transparent",
                hover_color=("gray82", "gray22"),
                text_color=("gray20", "gray80"),
                font=ctk.CTkFont(size=13),
                height=38,
                corner_radius=6,
                command=lambda nid=nav_id: self._switch(nid),
                **extra,
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
        panel_classes={
            "data": (TempPanel, {}),
            "config": (TempPanel, {}),
            "pipeline": (TempPanel, {}),
            "run": (TempPanel, {}),
            "results": (TempPanel, {}),
            "netron": (NetronPanel, {}),
        }
        for nav_id, (cls, kwargs) in panel_classes.items():
            panel=cls(self._content, state=self.state, **kwargs)
            panel.grid(row=0, column=0, sticky="nsew")
            self._panels[nav_id]=panel
        self.state.subscribe("stage_succeded", self._update_netron_badge)
        
    def _update_netron_badge(self, stage_id: str="", **_) -> None:
        n=sum(1 for v in self.state.netron_ready.values() if v)
        self._netron_badge.configure(text=f"    {n} checkpoint{'s' if n!=1 else ''} ready" if n else "")    
    
    def _switch(self, id: str) -> None:
        self._panels[id].tkraise()
        self._active=id
        active_fg=("gray75", "gray28")
        inactive_fg="transparent"
        for nid, btn in self._buttons.items():
            is_active=nid==id
            btn.configure(fg_color=active_fg if is_active else inactive_fg, font=ctk.CTkFont(size=13, weight="bold" if is_active else "normal"))       
    
    def _load_config(self) -> None:
        cfg_path=self.state.project_root/"config"/"pipeline.yaml"
        if cfg_path.exists():
            try:
                self.state.config_data=pc.load(cfg_path)
                self.state.config_path=Path("config/pieline.yaml")
                self.state.emit("config_loader")
            except Exception as e:
                print(f"[WARN] Could not load config: {e}")
        else:
            print(f"[INFO] Config not found at {cfg_path}. Set project root and reload in config panel.")