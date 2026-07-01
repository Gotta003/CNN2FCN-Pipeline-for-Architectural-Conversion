from __future__ import annotations
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Optional

class StageStatus(Enum):
    IDLE=auto()
    RUNNING=auto()
    SUCCESS=auto()
    FAILED=auto()
    SKIPPED=auto()
    
@dataclass
class StageResult:
    name: str
    status: StageStatus
    returncode: int=0
    error_msg: str=""

BASE_DIR="src/pipeline/stages"
    
STAGE_DEFS: list[tuple[str, str, str]]=[
    ("teacher", "Stage 0 - Teacher CNN", f"{BASE_DIR}/s0_teacher.py"),
    ("enforce", "Stage 1 - ENFORCE refiner", f"{BASE_DIR}/s1_enforce.py"),
    ("hint", "Stage 2 - HSR hint distillation", f"{BASE_DIR}/s2_hint.py"),
    ("dkd", "Stage 3 - DKD fine-tuning", f"{BASE_DIR}/s3_dkd.py"),
    ("anchor", "Stage 4 - 1M anchor student", f"{BASE_DIR}/s4_anchor.py"),
    ("nas", "Stage 5 - Evolutionary NAS", f"{BASE_DIR}/s5_nas.py"),
    ("eval", "Stage 6 - Evaluation", f"{BASE_DIR}/s6_eval.py"),
]

STAGE_DEPS: Dict[str, list[str]]={
    "teacher": [],
    "enforce": ["teacher"],
    "hint": ["teacher"],
    "dkd": ["teacher", "hint"],
    "anchor": ["teacher", "hint", "dkd"],
    "nas": ["teacher", "hint", "dkd", "anchor"],
    "eval": ["teacher", "hint", "dkd", "anchor", "nas"],
}

class StageRunner:
    def __init__(self, project_root: Path, config_path: Path, log_queue: queue.Queue, status_cb: Callable[[str, StageStatus], None]):
        self.project_root=project_root
        self.config_path=config_path
        self.log_queue=log_queue
        self.status_cb=status_cb
        self._abort=threading.Event()
        self._proc: Optional[subprocess.Popen]=None
        self._thread: Optional[threading.Thread]=None
        
    def run(self, enabled_stages: list[str])->None:
        if self._thread and self._thread.is_alive():
            return
        self._abort.clear()
        self._thread=threading.Thread(target=self._run_stages, args=(enabled_stages,), daemon=True)
        self._thread.start()
        
    def abort(self)->None:
        self._abort.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("[ABORT] Stage process terminated by user.")

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _log(self, msg: str) -> None:
        self.log_queue.put(msg)
        
    def _run_stages(self, enabled_stages: list[str]) -> None:
        pipeline_dir=Path(__file__).parent.parent
        for stage_id, display_name, script_rel in STAGE_DEFS:
            if self._abort.is_set():
                break
            if stage_id not in enabled_stages:
                self.status_cb(stage_id, StageStatus.SKIPPED)
                continue
            script=pipeline_dir/script_rel
            if not script.exists():
                self._log(f"[ERROR] Script not found: {script}")
                self.status_cb(stage_id, StageStatus.FAILED)
                break
            self._log(f"\n{'='*60}")
            self._log(f"    {display_name}")
            self._log(f"{'='*60}")
            self.status_cb(stage_id, StageStatus.RUNNING)

            cmd=[
                sys.executable, str(script),
                "--config", str(self.config_path),
                "--root", str(self.project_root),
            ]
            
            try:
                self._proc=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(self.project_root))
                for line in self._proc.stdout:
                    if self._abort.is_set():
                        break
                    self._log(line.rstrip())
                self._proc.wait()
                rc=self._proc.returncode
            except Exception as e:
                self._log(f"[ERROR] Failed to launch {display_name}: {e}")
                self.status_cb(stage_id, StageStatus.FAILED)
                break
            
            if self._abort.is_set():
                self.status_cb(stage_id, StageStatus.FAILED)
                break
            
            if rc==0:
                self._log(f"[OK] {display_name} completed successfully.")
                self.status_cb(stage_id, StageStatus.SUCCESS)
            else:
                self._log(f"[ERROR] {display_name} exited with code {rc}.")
                self.status_cb(stage_id, StageStatus.FAILED)
                break
        self._log("\n[PIPELINE] Run finished")