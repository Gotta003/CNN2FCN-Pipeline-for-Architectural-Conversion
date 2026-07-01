from __future__ import annotations
import tkinter as tk
import customtkinter as ctk

_TAGS: dict[str, str]={
    "ERROR": "#E24B4A",
    "WARN": "#EF9F27",
    "WARNING": "#EF9F27",
    "INFO": "#378ADD",
    "DEBUG": "#888888",
    "Ep ": "#1D9E75",
    "Gen ": "#7F77DD",
    "Improvement ": "#1DC48A",
    "Stage": "#C8A45A",
    "ABORT": "#E24B4A",
    "OK": "#1D9E75",
    "best": "#1D9E75",
    "%": "#EF9F27",
    "Download": "#378ADD",
    "Extracting": "#7F77DD",
    "Complete": "#1D9E75",
}

class LogViewer(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self._text=tk.Text(self, wrap="word", font=("Courier New", 11), state="disabled", bg="#1A1A1A", fg="#D4D4D4", relief="flat", borderwidth=0, padx=10, pady=8, selectbackground="#264F78")
        self._sb=ctk.CTkScrollbar(self, command=self._text.yview)
        self._text.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)
        for kw, col in _TAGS.items():
            self._text.tag_configure(kw, foreground=col)
            
    def append(self, line: str, replace: bool) -> None:
        self._text.configure(state="normal")
        if replace:
            try:
                self._text.delete("end-2l", "end-1l")
            except tk.TclError:
                pass
        tag=next((k for k in _TAGS if k in line), None)
        if tag:
            self._text.insert("end", line+"\n", tag)
        else:
            self._text.insert("end", line+"\n")
        self._text.see("end")
        self._text.configure(state="disabled")
        
    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")