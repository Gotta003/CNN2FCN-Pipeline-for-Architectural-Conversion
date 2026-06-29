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
from src.pipeline.models import TeacherCNN
torch.backends.cudnn.enabled = False

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
    dataset_path=root / str(data_cfg["dataset_path"])
    print(f"Loading dataset: {dataset_path}")
    ds=np.load(dataset_path)
    H, W=data_cfg["input_shape"][:2]
    X=ds[data_cfg["values_name"]][:,:H,:W]
    y=ds[data_cfg["labels_name"]].astype(int)
    
    min_label = y.min()
    max_label = y.max()
    num_unique_classes = len(np.unique(y))
    
    print(f"--- DIAGNOSTICA CLASSI ---")
    print(f"Label minima nel dataset: {min_label}")
    print(f"Label massima nel dataset: {max_label}")
    print(f"Numero di classi uniche rilevate: {num_unique_classes}")
    
    print(f"H: {H}, W: {W}, X_Shape: {X.shape}, Y_Shape: {y.shape}")
    #Remap labels
    unique=np.unique(y)
    mapping={c: i for i, c in enumerate(unique)}
    y=np.array([mapping[c] for c in y])
    n_classes=data_cfg["num_classes"]
    #Split
    test_frac=data_cfg["test_fraction"]
    val_frac=data_cfg["val_fraction"]
    X_tr, X_tmp, y_tr, y_tmp=train_test_split(X, y, test_size=test_frac, random_state=seed)
    X_va, X_te, y_va, y_te=train_test_split(X_tmp, y_tmp, test_size=val_frac, random_state=seed)
    print(f"Train: (X: {X_tr.shape}) (y: {y_tr.shape})")
    print(f"Val: (X: {X_va.shape}) (y: {y_va.shape})")
    print(f"Test: (X: {X_te.shape}) (y: {y_te.shape})")
    
    def to_tensor(arr, label=False):
        t=torch.LongTensor(arr) if label else torch.FloatTensor(arr)
        return t
    
    X_tr_t=to_tensor(X_tr).unsqueeze(1)
    X_va_t=to_tensor(X_va).unsqueeze(1)
    X_te_t=to_tensor(X_te).unsqueeze(1)
    y_tr_t=to_tensor(y_tr, label=True)
    y_va_t=to_tensor(y_va, label=True)
    y_te_t=to_tensor(y_te, label=True)
    batch=data_cfg["batch_size"]
    tr_loader=DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch, shuffle=True, drop_last=True)
    va_loader=DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=batch, shuffle=False)
    te_loader=DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch, shuffle=False)
    print(f"Train: {len(tr_loader)} Val: {len(va_loader)} Test: {len(te_loader)}")
    #Model
    model=TeacherCNN(input_shape=(1, H, W), num_classes=n_classes).to(device)
    print(f"Teacher params: {sum(p.numel() for p in model.parameters()):,}")
    opt=torch.optim.AdamW(model.parameters(), lr=teacher_cfg["lr"], weight_decay=teacher_cfg["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=teacher_cfg["epochs"], eta_min=1e-6)
    ce=nn.CrossEntropyLoss()
    best_val, best_state=0.0, None
    
    for epoch in range(teacher_cfg["epochs"]):
        model.train()
        nc, nt, ep_loss=0, 0, 0.0
        for xb, yb in tr_loader:
            xb, yb=xb.to(device), yb.to(device)
            opt.zero_grad()
            logits=model(xb)
            loss=ce(logits, yb)
            loss.backward()
            opt.step()
            ep_loss+=loss.item()
            nc+=logits.argmax(1).eq(yb).sum().item()
            nt+=yb.size(0)
        sched.step()
        
        model.eval()
        vc, vt=0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                vc+=model(xb.to(device)).argmax(1).eq(yb.to(device)).sum().item()
                vt+=yb.size(0)
        val_acc=vc/vt
        print(f"Ep {epoch+1:3d}/{teacher_cfg['epochs']} | loss={ep_loss/len(tr_loader):.4f} | train={nc/nt*100:.2f}% | val={val_acc*100:.2f}%")
        if val_acc>best_val:
            best_val=val_acc
            best_state={k: v.cpu() for k,v in model.state_dict().items()}
            out=root / teacher_cfg["weights_path"]
            torch.save(best_state, out)
            print(f" New best val={val_acc*100:.2f}% - save to {out}")
    print(f"\nStage 0 complete. Best val acc: {best_val*100:.2f}%")
    
if __name__=="__main__":
    main()