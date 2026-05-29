from __future__ import annotations
import threading
import webbrowser
from pathlib import Path
from typing import Optional
import customtkinter as ctk
from gui.state import AppState
from panels.base_panel import BasePanel

#Netron Integration
try:
    import netron
    HAS_NETRON=True
except ImportError:
    HAS_NETRON=False
    
# Embedded Browser Integration
try:
    from tkinterweb import HtmlFrame
    HAS_TKWEB=True
except ImportError:
    HAS_TKWEB=False
    
_NETRON_PORT=8080
_STAGE_PTH: dict[str, str]={
    "teacher": "kws_multi_cnn_model_pytorch.pth",
    "enforce": "refiner.pth",
    "hint":    "bridge_stage1.pth",
    "dkd":     "bridge_stage2.pth",
    "anchor":  "anchor_1M.pth",
}

class NetronPanel(BasePanel):
    """Sidebar panel that lets the user pick any stage checkpoint and launches a Neteron server"""
    def __init__(self, master, state: AppState, **kwargs):
        super().__init__(master, state, **kwargs)
        self.state=state
        self.configure(fg_color="transparent")
        self._server_thread: Optional[threading.Thread]=None
        self._server_running=False
        self._current_port: int=_NETRON_PORT
        self.build_content()
        
        state.subscribe("stage_succeeded", self._on_stage_succeded)
        state.subscribe("run_finished", self._on_run_finished)
    
    def build_content(self) -> None:
        ctk.CTkLabel(self, text="Netron Viewer", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
    
    # File List
    def _refresh_file_list(self) -> None:
        root=self.state.project_root
        found: list[str]=[]
        for s_id, fname in _STAGE_PTH.items():
            p=root/fname
            if p.exists():
                found.append(str(p))
        for p in sorted(root.glob("nas_bridge_*.pth")):
            found.append(str(p))
        for p in sorted(root.glob("*.pth")):
            s=str(p)
            if s not in found:
                found.append(s)
        labels=[Path(f).name for f in found] if found else ["- no .pth found -"]
        self._file_menu.configure(values=labels if found else ["- no .pth found -"])
        if found:
            self._file_menu.set(Path(found[0]).name)
            self._custom_entry.delete(0, "end")
            self._custom_entry.insert(0, found[0])
        self._found_paths={Path(f).name: f for f in found}
        
    def _on_file_selected(self, name: str) -> None:
        path=self._found_paths.get(name, "")
        self._custom_entry.delete(0, "end")
        self._custom_entry.insert(0, path)
        
    def _on_stage_succeded(self, stage_id: str="", **_) -> None:
        self._refresh_file_list()
        fname=_STAGE_PTH.get(stage_id)
        if fname and fname in self._found_paths:
            self._file_menu.set(fname)
            self._on_file_selected(fname)
        elif stage_id=="nas":
            nas=[n for n in self._found_paths if n.startswith("nas_bridge_")]
            if nas:
                self._file_menu.set(nas[0])
                self._on_file_selected(nas[0])
                
    def _on_run_finished(self, **_) -> None:
        self._refresh_file_list()