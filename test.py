#!/usr/bin/env python
"""
Confronta la valutazione del bridge_stage2 usando:
  (A) X_test_spatial -> forward_flat()   [esattamente il bug in _do_eval oggi]
  (B) X_test_full     -> forward_flat()  [la fix proposta]

Uso:
    python check_bridge_spatial_vs_flat.py --root /path/to/CNN2FCN-Pipeline-for-Architectural-Conversion
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
torch.backends.cudnn.enabled=False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="config/pipeline.yaml")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "pipeline" / "stages"))

    import config.config as pcfg
    from src.pipeline.data_utils import load_splits, DatasetNotPreparedError
    from src.pipeline.models import TeacherCNN, HSRBridge

    cfg = pcfg.load(root / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        splits = load_splits(cfg, root)
    except DatasetNotPreparedError as e:
        sys.exit(f"[ERRORE] {e}")

    n_classes = splits["n_classes"]
    UNKNOWN_SENTINEL = splits["unknown_sentinel"]
    known_indices = splits["known_indices"]
    td = cfg["data"]["tracker_dim"]
    s1_cfg = cfg.get("stage1", {})
    s2_cfg = cfg.get("stage2", {})

    print(f"n_classes={n_classes}  tracker_dim(config)={td}  pixel_dim={cfg['data']['pixel_dim']}")
    print(f"X_test_spatial shape: {tuple(splits['X_test_spatial'].shape)}")
    print(f"X_test_full   shape: {tuple(splits['X_test_full'].shape)}   (atteso: [N, pixel_dim+tracker_dim])")

    # ---- carica teacher + bridge stage2 -----------------------------
    teacher = TeacherCNN(input_shape=(1, 40, 40), num_classes=n_classes).to(device)
    teacher.load_state_dict(torch.load(root / cfg["teacher"]["weights_path"], map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    bridge = HSRBridge(
        teacher, input_shape=(1, 40, 40), tracker_dim=td,
        embed_dim=s1_cfg.get("embed_dim", 128),
        ranks=s1_cfg.get("hsr_ranks", [16, 8, 4]),
        dropout=s1_cfg.get("dropout", 0.4),
    ).to(device)
    bridge.classifier = nn.Linear(s1_cfg.get("embed_dim", 128), n_classes).to(device)
    weights_path = root / s2_cfg.get("weights_path", "models/bridge_stage2.pth")
    try:
        bridge.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"\nCaricato bridge_stage2 da {weights_path}: OK")
    except RuntimeError as e:
        print(f"\n[ERRORE caricamento bridge_stage2]: {e}")
        print("Se lo shape mismatch riguarda i tracker (fusion layer), il")
        print("tracker_dim nel config e' probabilmente diverso da quello con")
        print("cui il bridge e' stato allenato -> controlla tracker_dim nel")
        print("pipeline.yaml rispetto a quello usato nel notebook.")
        sys.exit(1)
    bridge.eval()
    for p in bridge.parameters():
        p.requires_grad = False

    def evaluate(x_tensor, label):
        loader = DataLoader(TensorDataset(x_tensor, splits["y_test"]), batch_size=64, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xf, lbl in loader:
                out, _ = bridge.forward_flat(xf.to(device))
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbl.numpy())
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        known_mask = labels_arr != UNKNOWN_SENTINEL
        acc = (preds_arr[known_mask] == labels_arr[known_mask]).mean() if known_mask.any() else 0.0
        print(f"\n[{label}] raw known-accuracy (argmax puro, no soglia): {acc*100:.2f}%")
        return acc

    print("\n" + "=" * 70)
    print("CONFRONTO: input SPATIAL (bug attuale) vs input FLAT (fix)")
    print("=" * 70)
    acc_bug = evaluate(splits["X_test_spatial"], "BUGGY: X_test_spatial -> forward_flat")
    acc_fix = evaluate(splits["X_test_full"], "FIX:   X_test_full -> forward_flat")

    print("\n" + "=" * 70)
    print("VERDETTO")
    print("=" * 70)
    if acc_fix - acc_bug > 0.20:
        print(f"  Confermato: passare X_test_spatial invece di X_test_full a forward_flat()")
        print(f"  fa crollare l'accuracy da {acc_fix*100:.1f}% a {acc_bug*100:.1f}%.")
        print("  La fix (usare X_test_full / X_val_full_flat in _do_eval) e' quella giusta.")
    else:
        print(f"  Differenza modesta ({acc_bug*100:.1f}% vs {acc_fix*100:.1f}%): il bug")
        print("  spatial/flat non è la causa principale per questo checkpoint,")
        print("  bisogna guardare altrove (es. tracker_dim mismatch, o checkpoint stale).")


if __name__ == "__main__":
    main()