"""
Shared state for pipeline to track the current process, like common struct
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

@dataclass
class AppState:
    project_root: Path=field(default_factory=Path.cwd)
    config_path: Path=field(default_factory=lambda: Path("config/pipeline.yaml"))
    config_data: Dict[str, Any]=field(default_factory=dict)
    stage_enabled: Dict[str, bool]=field(default_factory=lambda: {
        "teacher": True,
        "enforce": True,
        "hint": True,
        "dkd": True,
        "anchor": True,
        "nas": True,
        "eval": True,
    })
    is_running: bool=False
    netron_ready: Dict[str, bool]=field(default_factory=dict)
    #Events listeners
    _listeners: Dict[str, List[Callable]]=field(default_factory=dict)
    def subscribe(self, event: str, cb: Callable) -> None:
        self._listeners.setdefault(event, []).append(cb)
        
    def emit(self, event: str, **kwargs) -> None:
        for cb in self._listeners.get(event, []):
            try:
                cb(**kwargs)
            except Exception:
                pass
            
    def abs_config(self) -> Path:
        if self.config_path.is_absolute():
            return self.config_path
        return self.project_root/self.config_path