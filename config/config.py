from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

# Each entry: (section_key, display_label, field_descriptions)
SECTIONS: list[tuple[str, str, Dict[str, str]]] = [
    ("data", "Data", {
        "dataset_path":  "Path to .npz dataset file",
        "values_name":   "Key for feature array inside .npz",
        "labels_name":   "Key for label array inside .npz",
        "seed":          "Global random seed",
        "num_classes":   "Total number of classes (incl. unknown)",
        "batch_size":    "Mini-batch size for all stages",
        "test_fraction": "Fraction held out for test split",
        "val_fraction":  "Fraction of temp split used for validation",
    }),
    ("teacher", "Stage 0 — Teacher CNN", {
        "epochs":       "Training epochs",
        "lr":           "Learning rate",
        "weight_decay": "AdamW weight decay",
        "dropout":      "Dropout rate",
        "weights_path": "Output path for teacher .pth",
        "energy_finetune_epochs": "Epochs after training",
        "energy_finetune_lr": "Learning Rate of Fine Tuning",
        "energy_finetune_max_acc_drop": "Max acceptable negative difference between teacher and fine-tuner",
        "energy_weights_path": "Output path for fine tuner .pth",
        "energy_m_in": "Energy in input",
        "energy_m_out": "Energy in output",
        "energy_weight": "Importance score for loss function",
        "energy_ramp_epochs": "Constant increasing weighting of the function from start to that number",
    }),
    ("enforce", "Stage 1 — ENFORCE refiner", {
        "epochs":     "Pre-training epochs",
        "lr":         "Learning rate",
        "lambda_cls": "Classification head weight (≤0.1 keeps reconstruction dominant)",
        "batch_size": "Batch size for refiner",
    }),
    ("stage_hint", "Stage 2 — HSR hint distillation", {
        "epochs":         "Training epochs",
        "lr":             "Learning rate",
        "weight_decay":   "AdamW weight decay",
        "hint_weight":    "Weight on hybrid hint loss",
        "ce_weight":      "Weight on focal CE loss",
        "focal_gamma":    "Focal loss γ",
        "mixup_alpha":    "Mixup α parameter",
        "mixup_prob":     "Probability of applying mixup per batch",
        "patience":       "Early-stopping patience (epochs)",
        "embed_dim":      "Student embedding dimension",
        "dropout":        "Dropout rate in fusion head",
        "weights_path":   "Output path for Stage 1 bridge .pth",
    }),
    ("stage_dkd", "Stage 3 — DKD fine-tuning", {
        "epochs":         "Fine-tuning epochs",
        "lr":             "Learning rate",
        "weight_decay":   "AdamW weight decay",
        "kd_temperature": "Softmax temperature T for DKD",
        "dkd_alpha":      "TCKD loss weight α",
        "dkd_beta":       "NCKD loss weight β",
        "nuc_reg":        "Nuclear-norm penalty on U_r (μ)",
        "patience":       "Early-stopping patience (epochs)",
        "weights_path":   "Output path for Stage 2 bridge .pth",
    }),
    ("anchor", "Stage 4 — 1M anchor student", {
        "target_budget": "Parameter budget for the anchor (~1 000 000)",
        "epochs":        "Training epochs",
        "lr":            "Learning rate",
        "weight_decay":  "AdamW weight decay",
        "kd_temperature":"KD temperature T",
        "lambda_kd":     "KD loss weight (λ_kd)",
        "patience":      "Early-stopping patience (epochs)",
        "weights_path":  "Output path for anchor .pth",
    }),
    ("nas", "Stage 5 — Evolutionary NAS", {
        "generations":       "Number of evolutionary generations",
        "pop_size":          "Population size",
        "tournament_k":      "Tournament selection size k",
        "mutation_prob":     "Per-gene mutation probability",
        "proxy_epochs":      "Proxy training epochs per candidate",
        "proxy_lr":          "Proxy training learning rate",
        "finetune_epochs":   "Full fine-tune epochs for NAS winner",
        "finetune_lr":       "Fine-tune learning rate",
        "finetune_patience": "Fine-tune early-stopping patience",
        "kd_temperature":    "KD temperature for NAS training",
        "lambda_kd":         "KD loss weight",
        "results_path":      "JSON path for NAS search results",
        "subnet_prefix":     "Filename prefix for saved NAS subnets",
    }),
    ("eval", "Stage 6 — Evaluation", {
        "threshold_sweep_start": "Confidence threshold sweep start",
        "threshold_sweep_end":   "Confidence threshold sweep end",
        "threshold_sweep_step":  "Confidence threshold sweep step",
        "benchmark_plot_path":   "Output path for benchmark plot PNG",
    }),
]

LIST_FIELDS={"class_names", "input_shape", "hsr_ranks", "rank_choices", "fusion_choices", "embed_choices", "budgets"}

def load(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def save(cfg: Dict[str, Any], path: Path)->None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
def validate(cfg: Dict[str, Any])->list[str]:
    errors: list[str]=[]
    dataset=Path(cfg.get("data", {}).get("dataset_path", ""))
    if not dataset.exists():
        errors.append(f"Dataset not found: {dataset}")
    n_known=cfg.get("data", {}).get("num_classes", 0)
    n_all=len(cfg.get("data", {}).get("class_names_all", []))
    if n_known and n_all and n_known>=n_all:
        errors.append(f"num_classes ({n_known}) must be less than len(class_names_all) ({n_all})")
    return errors

def coerce_value(raw: str, original: Any) -> Any:
    if isinstance(original, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(original, int):
        try:
            return int(raw)
        except ValueError:
            return original
    if isinstance(original, float):
        try:
            return float(raw)
        except ValueError:
            return original
    if isinstance(original, list):
        elem_type=type(original[0]) if original else str
        parts=[p.strip() for p in raw.split(",") if p.strip()]
        try:
            return [elem_type(p) for p in parts]
        except ValueError:
            return original
    return raw