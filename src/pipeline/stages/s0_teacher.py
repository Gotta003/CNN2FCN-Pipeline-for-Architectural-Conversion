from __future__ import annotations
import argparse
import sys
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
project_root=str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)
from src.pipeline.models import TeacherCNN
torch.backends.cudnn.enabled = False

import functools
print=functools.partial(print, flush=True)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    return p.parse_args()

def main():
    args=parse_args()
    root=Path(args.root).resolve()
    sys.path.insert(0, str(root))
    with open(args.config) as f:
        cfg=yaml.safe_load(f)
    data_cfg=cfg["data"]
    teacher_cfg=cfg["teacher"]
    seed=data_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    #Dataset Loading
    from src.pipeline.data_utils import load_splits, DatasetNotPreparedError
    try:
        splits=load_splits(cfg, root)
    except DatasetNotPreparedError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    n_classes=splits["n_classes"]
    H, W=data_cfg["input_shape"][:2]    
    batch=data_cfg["batch_size"]
    tr_loader=DataLoader(TensorDataset(splits["X_train_spatial"], splits["y_train"]), batch_size=batch, shuffle=True, drop_last=True)
    va_loader=DataLoader(TensorDataset(splits["X_val_known_spatial"], splits["y_val_known"]), batch_size=batch, shuffle=False)
    print(f"Train: {len(tr_loader)} Val: {len(va_loader)}")
    #Model
    model=TeacherCNN(input_shape=(1, H, W), num_classes=n_classes).to(device)
    print(f"Teacher params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train: {len(splits['y_train']):,}  Val: {len(splits['y_val_known']):,}  Test: {len(splits['y_test']):,}  (fixed splits)")
    opt=torch.optim.AdamW(model.parameters(), lr=teacher_cfg["lr"], weight_decay=teacher_cfg["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=teacher_cfg["epochs"], eta_min=1e-6)
    ce=nn.CrossEntropyLoss(weight=splits["class_weights"].to(device).float())
    best_val, best_state=0.0, None
    from src.pipeline.metrics_logger import MetricsLogger
    logger=MetricsLogger(root, 'teacher')
    
    from sklearn.metrics import f1_score
    for epoch in range(teacher_cfg["epochs"]):
        model.train()
        nc, nt, ep_loss=0, 0, 0.0
        train_preds, train_targets=[], []
        for xb, yb in tr_loader:
            xb, yb=xb.to(device), yb.to(device)
            opt.zero_grad()
            logits=model(xb)
            loss=ce(logits, yb)
            loss.backward()
            opt.step()
            ep_loss+=loss.item()
            preds=logits.argmax(1)
            nc+=preds.eq(yb).sum().item()
            nt+=yb.size(0)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())
        sched.step()
        
        model.eval()
        vc, vt, val_loss_sum=0, 0, 0.0
        val_preds, val_targets=[], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb=xb.to(device), yb.to(device)
                logits=model(xb)
                loss=ce(logits, yb)
                val_loss_sum+=loss.item()
                preds=logits.argmax(1)
                vc+=preds.eq(yb).sum().item()
                vt+=yb.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())
        train_loss=ep_loss/len(tr_loader)
        train_acc=nc/nt
        train_f1=float(f1_score(train_targets, train_preds, average="micro"))
        val_loss=val_loss_sum/len(va_loader)
        val_acc=vc/vt
        val_f1=float(f1_score(val_targets, val_preds, average="macro"))
        print(f"Ep {epoch+1:3d}/{teacher_cfg['epochs']} | Train: loss={train_loss:.4f} acc={train_acc*100:.2f}% f1={train_f1*100:.4f}% | Val: loss={val_loss:.4f} acc={val_acc*100:.2f}% f1={val_f1*100:.4f}%")
        logger.log_epoch(epoch+1, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)
        if val_acc>best_val:
            best_val=val_acc
            best_state={k: v.cpu() for k,v in model.state_dict().items()}
            out=root / teacher_cfg["weights_path"]
            torch.save(best_state, out)
            print(f" New best val={val_acc*100:.2f}% - save to {out}")
    logger.finalize()
    print(f"\nStage 0 complete. Best val acc: {best_val*100:.2f}%")
    
if __name__=="__main__":
    main()