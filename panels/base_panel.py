"""
Parent class of a generic panel window in the main content space in app.py
"""
from __future__ import annotations
from typing import Optional
import customtkinter as ctk
from gui.state import AppState

class BasePanel(ctk.CTkFrame):
    """
    Abstract class for all GUI panels. Standardized layout, typography, margin, header and state

    Args:
        master: (Generic Parent component in which the Panel will be displayed),
        state: AppState (Current status of the application in dataclass type)
        **kwargs ()
    """
    def __init__(self, master, state:AppState, **kwargs):
        super().__init__(master, **kwargs)
        self.state=state
        self.configure(fg_color="transparent")
        self.header_frame: Optional[ctk.CTkFrame]=None
        self.action_frame: Optional[ctk.CTkFrame]=None
        self.status_lbl: Optional[ctk.CTkLabel]=None
        self.main_container: Optional[ctk.CTkFrame]=None
        self.build_content()
        
    def set_status(self, msg: str, is_error: bool=False, is_success: bool=False):
        """Thread status text adjustments with color"""
        def _update():
            if is_error:
                color="#E24B4A"
            elif is_success or msg.startswith("✓") or msg.startswith("⬡"):
                color="#1D9E75"
            else:
                color=("gray50", "gray55")
            self.status_lbl.configure(text=msg, text_color=color)
        self.after(0, _update)
        
    def build_content(self) -> None:
        """Abstract method. Derived classes must override this to insert their components"""
        pass