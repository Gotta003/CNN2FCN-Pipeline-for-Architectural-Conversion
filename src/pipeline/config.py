"""
Load, validate and save pipeline.yaml, default load
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

SECTION: list[tuple[str, str, Dict[str, str]]]=[
    ("data", "Data", {
        
    }),
    ("teacher", "Stage 0 - Teacher CNN", {
        
    }),
    ("enforce", "Stage 1 - ENFORCE Refiner", {
        
    }),
    ("stage_hint", "Stage 2 - HSR hint Distillation", {
        
    }),
    ("stage_dkd", "Stage 3 - DKD Fine Tuning", {
        
    }),
    ("anchor", "Stage 4 - 1M anchor Student", {
        
    }),
    ("nas", "Stage 5 - Evolutionary NAS", {
        
    }),
    ("eval", "Stage 6 - Evaluation", {
        
    }),
]

def load(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)