"""
Parent class of a temporary panel window in the main content space in app.py
"""
from __future__ import annotations
from typing import Optional
import customtkinter as ctk
from gui.state import AppState
from gui.panels.base_panel import BasePanel
from src.assets import Icons

class TempPanel(BasePanel):
    """Placeholder viewport panel used as a fallback while implementing the others"""
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        
    def build_content(self) -> None:
        dev_card=ctk.CTkFrame(self.main_container, fg_color=("gray94", "gray14"), corner_radius=10)
        #dev_card.pack(fill="both", expand=True, padx=24, pady=(10, 24))
        #Box Info
        inner_layout=ctk.CTkFrame(dev_card, fg_color="transparent")
        inner_layout.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner_layout, text="", image=Icons.hexagon_warning).pack(pady=(0,8))
        ctk.CTkLabel(inner_layout, text="PANEL IN DEVELOPMENT", font=ctk.CTkFont(family="Courier New", size=16, weight="bold"), text_color=("gray30", "gray70")).pack()
        ctk.CTkLabel(inner_layout, text="Pipeline hooks are active. Layout definitions are in implementation", font=ctk.CTkFont(size=12), text_color=("gray50", "gray55"), justify="center").pack(pady=(6,0))
        self.set_status("System Alert: Viewing unconfigured module stub.")