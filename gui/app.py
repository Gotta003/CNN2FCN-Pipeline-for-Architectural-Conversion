"""
Root window with sidebar navigation that manipulates the main windows
"""

from __future__ import annotations
from pathlib import Path
import customtkinter as ctk
from gui.state import AppState
import src.pipeline.config as pc

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self, project_root: Path):
        super().__init__()
        self.title("CNN2FCN - ENFORCE + EvoNAS Pipeline")
        self.geometry("1200x780")
        self.minsize(900, 620)
        self.state=AppState(project_root=project_root)
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
        print(self.state)
        
