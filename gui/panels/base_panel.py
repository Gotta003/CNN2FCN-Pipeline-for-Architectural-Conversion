from __future__ import annotations
from typing import Optional
import customtkinter as ctk
from gui.state import AppState

class BasePanel(ctk.CTkFrame):
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, **kwargs)
        self.app_state=app_state
        self.configure(fg_color="transparent")
        self.header_frame: Optional[ctk.CTkFrame]=None
        self.action_frame: Optional[ctk.CTkFrame]=None
        self.status_lbl: Optional[ctk.CTkLabel]=None
        self.main_container: Optional[ctk.CTkFrame]=None
        self.status_lbl = ctk.CTkLabel(
            self, text="", anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray55")
        )
        self.status_lbl.pack(fill="x", padx=24, pady=(2, 6))
        self.build_content()
        
    def set_status(self, msg: str, is_error: bool=False, is_success: bool=False):
        """Thread status text adjustments with color"""
        def _update():
            if is_error:
                color="#E24B4A"
            elif is_success:
                color="#1D9E75"
            else:
                color=("gray50", "gray55")
            self.status_lbl.configure(text=msg, text_color=color)
        self.after(0, _update)
        
    def build_content(self) -> None:
        """Abstract method. Derived classes must override this to insert their components"""
        pass