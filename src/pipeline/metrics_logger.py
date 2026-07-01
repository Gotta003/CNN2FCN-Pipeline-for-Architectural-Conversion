from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

class MetricsLogger:
    def __init__(self, root: Path, stage_name: str):
        self.path=root / "models"/f"metrics_{stage_name}.json"
        self._data: dict={
            "stage": stage_name,
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "summary": {},
        }
        self._best_val_f1=0.0
        self._best_val_acc=0.0
        self._best_epoch=0
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float, train_f1: Optional[float]=None, val_f1: Optional[float]=None) -> None:
        if train_f1 is None:
            train_f1=train_acc
        if val_f1 is None:
            val_f1=val_acc
        self._data["epochs"].append(epoch)
        self._data["train_loss"].append(round(train_loss, 6))
        self._data["val_loss"].append(round(val_loss, 6))
        self._data["train_acc"].append(round(train_acc, 6))
        self._data["val_acc"].append(round(val_acc, 6))
        self._data["train_f1"].append(round(train_f1, 6))
        self._data["val_f1"].append(round(val_f1, 6))
        if val_f1>self._best_val_f1:
            self._best_val_f1=val_f1
            self._best_epoch=epoch
        if val_acc>self._best_val_acc:
            self._best_val_acc=val_acc
        self._flush()
        
    def finalize(self) -> None:
        d=self._data
        n=len(d["epochs"])
        if n==0:
            return
        self._data["summary"]={
            "final_train_loss": d["train_loss"][-1],
            "final_val_loss": d["val_loss"][-1],
            "final_train_acc": d["train_acc"][-1],
            "final_val_acc": d["val_acc"][-1],
            "final_train_f1": d["train_f1"][-1],
            "final_val_f1": d["val_f1"][-1],
            "best_val_f1": self._best_val_f1,
            "best_val_acc": self._best_val_acc,
            "best_epoch": self._best_epoch,
        }
        self._flush()
        
    def _flush(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass