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
    dataset_ready: bool=False
    dataset_manifest: Optional[Dict[str, Any]]=None
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
    
    def check_dataset_ready(self) -> bool:
        import json
        import hashlib
        required=["dataset_train.npz", "dataset_val_known.npz", "dataset_val_full.npz","dataset_test.npz"]
        if not all((self.project_root/"data"/f).exists() for f in required):
            self.dataset_ready=False
            self.dataset_manifest=None
            return False
        manifest_path=self.project_root/"data"/"dataset_manifest.json"
        if not manifest_path.exists():
            self.dataset_ready=False
            self.dataset_manifest=None
            return False
        try:
            with open(manifest_path) as f:
                manifest=json.load(f)
        except Exception:
            self.dataset_ready=False
            self.dataset_manifest=None
            return False
        
        data_cfg=self.config_data.get("data", {})
        src_rel=data_cfg.get("dataset_path", "")
        src_path=self.project_root/src_rel if src_rel else None
        if src_path and src_path.exists():
            try:
                h=hashlib.sha256()
                with open(src_path, "rb") as f:
                    while chunk:=f.read(1<<20):
                        h.update(chunk)
                current_hash=h.hexdigest()[:16]
                if current_hash!=manifest.get("source_hash"):
                    self.dataset_ready=False
                    self.dataset_manifest=manifest
                    return False
            except Exception:
                pass
        self.dataset_ready=True
        self.dataset_manifest=manifest
        return True