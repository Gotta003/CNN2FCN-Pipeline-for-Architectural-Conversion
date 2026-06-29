from __future__ import annotations
from typing import Any, Callable, Dict
import customtkinter as ctk

class YamlForm(ctk.CTkScrollableFrame):
    def __init__(self, master, sections, config_data: dict, on_change: Callable[[str, str, str], None], **kwargs):
        super().__init__(master, **kwargs)
        self._on_change=on_change
        self._entries: Dict[tuple, ctk.CTkEntry]={}
        self._build(sections, config_data)
        
    def _build(self, sections, config_data: dict) -> None:
        for sec_key, sec_label, descs in sections:
            sec_data=config_data.get(sec_key, {})
            header=ctk.CTkLabel(self, text=sec_label, font=ctk.CTkFont(size=13, weight="bold"), anchor="w")
            header.pack(fill="x", padx=16, pady=(14, 4))
            sep=ctk.CTkFrame(self, height=1, fg_color=("gray75", "gray35"))
            sep.pack(fill="x", padx=16, pady=(0, 8))
            for key, desc in descs.items():
                raw=sec_data.get(key, "")
                if key not in sec_data:
                    continue
                row=ctk.CTkFrame(self, fg_color="transparent")
                row.pack(fill="x", padx=16, pady=3)
                row.columnconfigure(1, weight=1)
                lbl=ctk.CTkLabel(row, text=key, anchor="w", font=ctk.CTkFont(size=12), width=160, text_color=("gray20", "gray85"))
                lbl.grid(row=0, column=0, sticky="w", padx=(0, 8))
                tip=ctk.CTkLabel(row, text=desc, anchor="w", font=ctk.CTkFont(size=10), text_color=("gray50", "gray55"))
                tip.grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 8))
                entry=ctk.CTkEntry(row, font=ctk.CTkFont(family="Courier New", size=12))
                entry.insert(0, self._display(raw))
                entry.grid(row=0, column=1, sticky="ew")
                entry.bind("<KeyRelease>", lambda e, s=sec_key: self._on_change(s, key, e.widget.get())) 
                self._entries[(sec_key, key)]=entry
                
    @staticmethod
    def _display(val: Any) -> str:
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)