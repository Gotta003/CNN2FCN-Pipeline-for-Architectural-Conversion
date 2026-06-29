from __future__ import annotations
import threading
import webbrowser
from pathlib import Path
from typing import Optional
import customtkinter as ctk
from gui.state import AppState
from panels.base_panel import BasePanel
from src.assets import Icons

# Netron Integration
try:
    import netron
    HAS_NETRON=True
except ImportError:
    HAS_NETRON=False
    
# Embedded Browser Integration (Utilizzato come flag per mostrare i messaggi di supporto)
try:
    import webview
    HAS_WEBVIEW=True
except ImportError:
    HAS_WEBVIEW=False
    
_NETRON_PORT=8080
_STAGE_PTH: dict[str, str]={
    "teacher": "kws_multi_cnn_model_pytorch.pth",
    "enforce": "refiner.pth",
    "hint":    "bridge_stage1.pth",
    "dkd":     "bridge_stage2.pth",
    "anchor":  "anchor_1M.pth",
}

class NetronPanel(BasePanel):
    """Sidebar panel that lets the user pick any stage checkpoint and launches a Netron server"""
    def __init__(self, master, app_state: AppState, **kwargs):
        super().__init__(master, app_state, **kwargs)
        self.state=app_state
        self.configure(fg_color="transparent")
        self._server_thread: Optional[threading.Thread]=None
        self._server_running=False
        self._current_port: int=_NETRON_PORT
        
        # Riferimenti per tracciare la finestra di visualizzazione nativa PyQt5
        self._qt_app=None
        self._qt_window=None
        
        app_state.subscribe("stage_succeeded", self._on_stage_succeded)
        app_state.subscribe("run_finished", self._on_run_finished)
    
    def build_content(self)->None:
        ctk.CTkLabel(self, text="Netron Viewer", font=ctk.CTkFont(size=20, weight="bold"), anchor="w").pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(self, text="Visualize model architecture from any stage checkpoint.", font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), anchor="w").pack(fill="x", padx=24, pady=(0, 12))
        
        sep=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep.pack(fill="x", padx=24, pady=(0, 16))
        
        # File Selector
        sel_frame=ctk.CTkFrame(self, fg_color="transparent")
        sel_frame.pack(fill="x", padx=24, pady=(0, 8))
        sel_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(sel_frame, text="Checkpoint", font=ctk.CTkFont(size=12), width=90, anchor="w").grid(row=0, column=0, sticky="w")
        
        self._file_var=ctk.StringVar(value="select")
        self._file_menu=ctk.CTkOptionMenu(sel_frame, variable=self._file_var, values=["select"], command=self._on_file_selected, dynamic_resizing=False)
        self._file_menu.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ctk.CTkButton(sel_frame, text="", image=Icons.refresh, width=36, command=self._refresh_file_list).grid(row=0, column=2, sticky="e")
        
        # Custom Path Row
        path_frame=ctk.CTkFrame(self, fg_color="transparent")
        path_frame.pack(fill="x", padx=24, pady=(0, 12))
        path_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(path_frame, text="Or browse", font=ctk.CTkFont(size=12), width=90, anchor="w").grid(row=0, column=0, sticky="w")
        
        self._custom_entry=ctk.CTkEntry(path_frame, font=ctk.CTkFont(family="Courier New", size=11), placeholder_text="path/to/model.pth")
        self._custom_entry.grid(row=0, column=1, sticky="ew", padx=(8,8))
        ctk.CTkButton(path_frame, text="Browse", width=70, command=self._browse).grid(row=0, column=2, sticky="e")
        
        # Port row
        port_frame=ctk.CTkFrame(self, fg_color="transparent")
        port_frame.pack(fill="x", padx=24, pady=(0, 12))
        ctk.CTkLabel(port_frame, text="Port", font=ctk.CTkFont(size=12), width=90, anchor="w").pack(side="left")
        
        self._port_entry=ctk.CTkEntry(port_frame, width=80, font=ctk.CTkFont(size=12))
        self._port_entry.insert(0, str(_NETRON_PORT))
        self._port_entry.pack(side="left", padx=(8,0))
        
        # Buttons
        btn_row=ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(0, 12))
        
        self._launch_btn=ctk.CTkButton(btn_row, text="Launch Netron", image=Icons.hexagon, width=150, command=self._launch)
        self._launch_btn.pack(side="left", padx=(0,8))
        
        self._stop_btn=ctk.CTkButton(btn_row, text="Stop Server", image=Icons.stop, width=130, fg_color="#C0392B", hover_color="#922B21", state="disabled", command=self._stop)
        self._stop_btn.pack(side="left", padx=(0,8))
        
        self._open_browser_btn=ctk.CTkButton(btn_row, text="Open in browser", image=Icons.redirect, width=150, fg_color="transparent", border_width=1, state="disabled", command=self._open_browser)
        self._open_browser_btn.pack(side="left")
        
        # Status Bar
        self._status_lbl=ctk.CTkLabel(self, text="", anchor="w", font=ctk.CTkFont(size=11), text_color=("gray50", "gray55"))
        self._status_lbl.pack(fill="x", padx=24, pady=(0, 8))
        
        sep2=ctk.CTkFrame(self, height=1, fg_color=("gray80", "gray30"))
        sep2.pack(fill="x", padx=24, pady=(0, 0))
        
        # Center Info Panel Container
        self._browser_container=ctk.CTkFrame(self, fg_color=("gray94", "gray14"), corner_radius=8)
        self._browser_container.pack(fill="both", expand=True, padx=24, pady=16)
        
        self._prompt_inner=ctk.CTkFrame(self._browser_container, fg_color="transparent")
        self._prompt_inner.place(relx=0.5, rely=0.5, anchor="center")

        self._center_icon=ctk.CTkLabel(self._prompt_inner, text="⬡", font=ctk.CTkFont(size=44), text_color=("gray40", "gray50"))
        self._center_icon.pack(pady=(0, 8))

        self._center_msg=ctk.CTkLabel(
            self._prompt_inner,
            text="Select a checkpoint above and click\nLaunch Netron to open the interactive graph workspace.",
            font=ctk.CTkFont(size=12), text_color=("gray40", "gray60"), justify="center"
        )
        self._center_msg.pack()
        
        self._refresh_file_list()
        
    def _open_browser(self):
        webbrowser.open(f"http://localhost:{self._current_port}")
        
    def _stop(self):
        """Disattiva il server e distrugge la finestra nativa di Chromium"""
        if self._qt_window and hasattr(self, '_automation_signals'):
            try:
                self._automation_signals.hide_window.emit()
            except Exception:
                pass

        self._stop_server_silent()
        self._set_status("Server Stopped")
        self._stop_btn.configure(state="disabled")
        self._open_browser_btn.configure(state="disabled")
        self._center_msg.configure(text="Select a checkpoint above and click\nLaunch Netron to open the interactive graph workspace.")
    
    def _handle_manual_window_close(self):
        """Intercetta la chiusura asincrona della finestra X del browser"""
        if self._server_running:
            self._stop()

    def _set_status(self, msg: str, is_error: bool=False, is_success: bool=False, image=None)->None:
        if is_error:
            color="#E24B4A"
            icon=None
        elif is_success:
            color="#1D9E75"
            icon=Icons.check
        else:
            color=("gray50", "gray55")
            icon=None
        if image is not None:
            icon=image
        display_text=f" {msg}" if icon else msg
        self._status_lbl.configure(text=display_text, text_color=color, image=icon, compound="left" if icon else "none")
        
    def _launch(self)->None:
        if not HAS_NETRON:
            self._set_status("netron not installed. pip install netron", is_error=True)
            return
        path=self._resolve_path()
        if path is None:
            return
        port=self._resolve_port()
        self._current_port=port
        if self._server_running:
            self._stop_server_silent()
        self._set_status(f"Starting Netron on port {port}...", is_success=True)
        self._launch_btn.configure(state="disabled")
        
        def _serve():
            try:
                netron.start(str(path), address=("localhost", port), browse=False)
            except Exception as e:
                self.after(0, self._set_status, f"Netron error: {e}", True)
        
        self._server_thread=threading.Thread(target=_serve, daemon=True)
        self._server_thread.start()
        self._server_running=True
        self.after(1200, self._on_server_ready, path, port)
        
    def _on_server_ready(self, path: Path, port: int)->None:
        url=f"http://localhost:{port}"
        self._set_status(f"Netron running - {path.name}->{url}", is_error=False, image=Icons.hexagon)
        self._stop_btn.configure(state="normal")
        self._open_browser_btn.configure(state="normal")
        self._launch_btn.configure(state="normal")
        self._center_msg.configure(text=f"Active Workspace Window Open\nViewing: {path.name}")
        
        if self._qt_window and hasattr(self, '_automation_signals'):
            self._automation_signals.update_url.emit(url, f"Netron Architecture Viewer - {path.name}")
            return
        
        def launch_pure_qt_browser():
            import sys
            import os
            os.environ["QTWEBENGINE_DISABLE_SANDBOX"]="1"
            
            from PyQt5.QtWidgets import QApplication, QMainWindow
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtCore import QUrl, Qt, QObject, pyqtSignal
            
            class BrowserSignals(QObject):
                update_url=pyqtSignal(str, str)
                hide_window=pyqtSignal()
            
            self._automation_signals=BrowserSignals()
            self._qt_app=QApplication.instance()
            if not self._qt_app:
                self._qt_app=QApplication([sys.argv[0], "--disable-web-security"])
                
            self._qt_window=QMainWindow()
            self._qt_window.setWindowTitle(f"Netron Architecture Viewer - {path.name}")
            self._qt_window.resize(1100, 750)
            self._qt_window.setAttribute(Qt.WA_DeleteOnClose, False)
            browser=QWebEngineView()
            browser.setUrl(QUrl(url))
            self._qt_window.setCentralWidget(browser)
            
            def slot_update_url(new_url, new_title):
                self._qt_window.setWindowTitle(new_title)
                browser.setUrl(QUrl(new_url))
                self._qt_window.show()
                self._qt_window.raise_()
                
            def slot_hide_window():
                self._qt_window.hide()
                
            self._automation_signals.update_url.connect(slot_update_url)
            self._automation_signals.hide_window.connect(slot_hide_window)
            
            def custom_close_event(event):
                self.after(0, self._handle_manual_window_close)
                event.ignore()
                self._qt_window.hide()
                
            self._qt_window.closeEvent=custom_close_event
            self._qt_window.show()
            self._qt_app.exec_()

        browser_thread=threading.Thread(target=launch_pure_qt_browser, daemon=True)
        browser_thread.start()

    def _stop_server_silent(self)->None:
        try:
            if HAS_NETRON:
                netron.stop()
        except Exception:
            pass
        self._server_running=False
        
    def _resolve_port(self)->int:
        try:
            return int(self._port_entry.get().strip())
        except ValueError:
            return _NETRON_PORT
        
    def _resolve_path(self)->Optional[Path]:
        raw=self._custom_entry.get().strip()
        if not raw:
            self._set_status("No checkpoint selected", is_error=True)
            return None
        p=Path(raw)
        if not p.is_absolute():
            p=self.app_state.project_root / p
        if not p.exists():
            self._set_status(f"File not found: {p}", is_error=True)
            return None
        return p
    
    def _browse(self)->None:
        from tkinter import filedialog
        path=filedialog.askopenfilename(title="Select model checkpoint", filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All", "*.*")], initialdir=str(self.app_state.project_root),)
        if path:
            self._custom_entry.delete(0, "end")
            self._custom_entry.insert(0, path)
    
    def _refresh_file_list(self)->None:
        root=Path(self.app_state.project_root, "models")
        found: list[str]=[]
        for s_id, fname in _STAGE_PTH.items():
            p=root / fname
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
        
    def _on_file_selected(self, name: str)->None:
        path=self._found_paths.get(name, "")
        self._custom_entry.delete(0, "end")
        self._custom_entry.insert(0, path)
        
    def _on_stage_succeded(self, stage_id: str="", **_)->None:
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
                
    def _on_run_finished(self, **_)->None:
        self._refresh_file_list()