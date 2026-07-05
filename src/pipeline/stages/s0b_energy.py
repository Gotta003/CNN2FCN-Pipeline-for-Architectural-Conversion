from pathlib import Path
import argparse
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset 
project_root=str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)
from src.pipeline.models import TeacherCNN
torch.backends.cudnn.enabled=False

import functools
print=functools.partial(print, flush=True)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    return p.parse_args()

@torch.no_grad()
def compute_energy_stats(model, loader_known, loader_unknown, device, temperature=1.0):
    model.eval()
    stats={}
    kn_e=[]
    unk_e=[]
    for xb, _ in loader_known:
        logits=model(xb.to(device))
        e=-temperature*torch.logsumexp(logits/temperature, dim=1)
        kn_e.extend(e.cpu().numpy())
    for (xb,) in loader_unknown:
        logits=model(xb.to(device))
        e=-temperature*torch.logsumexp(logits/temperature, dim=1)
        unk_e.extend(e.cpu().numpy())
    kn_e=np.array(kn_e)
    unk_e=np.array(unk_e)
    stats={
        "kn_mean": float(kn_e.mean()) if len(kn_e) else float("nan"),
        "unk_mean": float(unk_e.mean()) if len(unk_e) else float("nan")
    }
    stats["gap"]=stats["kn_mean"]-stats["unk_mean"] if len(kn_e) and len(unk_e) else float("nan")
    stats["overlap_at_kn_mean"]=float((unk_e<stats["kn_mean"]).mean()) if len(kn_e) and len(unk_e) else float("nan")
    return stats

@torch.no_grad()
def evaluate_full(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval()
    vc, vt, loss_sum=0, 0, 0.0
    all_preds, all_targets=[], []
    for xb, yb in loader:
        xb, yb=xb.to(device), yb.to(device)
        logits=model(xb)
        loss=F.cross_entropy(logits, yb)
        loss_sum+=loss.item()
        preds=logits.argmax(1)
        vc+=preds.eq(yb).sum().item()
        vt+=yb.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())
    acc=vc/vt if vt else 0.0
    f1=float(f1_score(all_targets, all_preds, average="macro")) if vt else 0.0
    val_loss=loss_sum/len(loader) if len(loader) else 0.0
    return val_loss, acc, f1

@torch.no_grad()
def evaluate_acc(model, loader, device):
    _, acc, _=evaluate_full(model, loader, device)
    return acc

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
    
    from src.pipeline.data_utils import load_splits, DatasetNotPreparedError
    try:
        splits=load_splits(cfg, root)
    except DatasetNotPreparedError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    n_classes=splits["n_classes"]
    H, W=data_cfg["input_shape"][:2]
    batch=data_cfg["batch_size"]
    tr_loader=DataLoader(TensorDataset(splits["X_train_spatial_k"], splits["y_train"]), batch_size=batch, shuffle=True, drop_last=True)
    ood_loader=DataLoader(TensorDataset(splits["X_train_spatial_u"]), batch_size=batch, drop_last=True)
    ood_eval_loader=DataLoader(TensorDataset(splits["X_train_spatial_u"]), batch_size=batch, shuffle=False)
    va_loader=DataLoader(TensorDataset(splits["X_val_spatial_k"], splits["y_val"]), batch_size=batch, shuffle=False)
    va_unk_loader=None
    if "X_val_spatial_u" in splits and splits["X_val_spatial_u"].size(0)>0:
        va_unk_loader=DataLoader(TensorDataset(splits["X_val_spatial_u"]), batch_size=batch, shuffle=False)
    unk_monitor_loader=va_unk_loader if va_unk_loader is not None else ood_eval_loader
    
    src_weights=root/teacher_cfg["weights_path"]
    if not src_weights.exists():
        print(f"[ERROR] No checkpoint teacher found in {src_weights}. Execute Stage 0 first")
        sys.exit(1)
    model=TeacherCNN(input_shape=(1, H, W), num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(src_weights, map_location=device))
    print(f"Load teacher from {src_weights}")
    base_val_acc=evaluate_acc(model, va_loader, device)
    base_stats=compute_energy_stats(model, va_loader, unk_monitor_loader, device)
    print(f"Before fine-tuning: val_acc={base_val_acc*100:.2f}% kn_mean={base_stats["kn_mean"]:.3f} unk_mean={base_stats["unk_mean"]:.3f} gap={base_stats["gap"]:.3f} overlap={base_stats["overlap_at_kn_mean"]*100:.1f}%")
    
    epochs=teacher_cfg.get("energy_finetune_epochs", 15)
    lr=teacher_cfg.get("energy_finetune_lr", 1e-4),
    m_in=teacher_cfg.get("energy_m_in", -6.0)
    m_out=teacher_cfg.get("energy_m_out", 0.0)
    energy_weight=teacher_cfg.get("energy_weight", 0.3)
    temperature=teacher_cfg.get("energy_temperature", 1.0)
    ramp_epochs=teacher_cfg.get("energy_ramp_epochs", 3)
    max_acc_drop=teacher_cfg.get("energy_finetune_max_acc_drop", 0.02)
    out_path=root/teacher_cfg.get("energy_weights_path", "models/kws_multi_cnn_model_pytorch_energy.pth")
    
    print(f"Fine-tuning: epochs={epochs} lr={lr} m_in={m_in} m_out={m_out} weight={energy_weight} ramp={ramp_epochs} max_acc_drop={max_acc_drop*100:.1f}%")
    print(f"Output: {out_path}")
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=teacher_cfg.get("weight_decay", 1e-4))
    best_gap=1e-9
    best_state=None
    best_epoch=-1
    
    from src.pipeline.metrics_logger import MetricsLogger
    logger=MetricsLogger(root, "teacher_energy_finetune")
    from sklearn.metrics import f1_score
    for epoch in epochs:
        model.train()
        current_weight=energy_weight*min(1.0, (epoch+1)/max(1, ramp_epochs))
        ood_iter=iter(ood_loader)
        ep_ce=0.0
        ep_energy=0.0
        nb=0
        nc=0
        nt=0
        train_preds=[]
        train_targets=[]
        for xb, yb in tr_loader:
            xb, yb=xb.to(device), yb.to(device)
            opt.zero_grad()
            logits=model(xb)
            ce_loss=F.cross_entropy(logits, yb)
            energy_in=-temperature*torch.logsumexp(logits/temperature, dim=1)
            loss_energy_in=torch.pow(F.relu(energy_in-m_in), 2).mean()
            try:
                xb_out=next(ood_iter)[0].to(device)
            except StopIteration:
                ood_iter=iter(ood_loader)
                xb_out=next(ood_iter)[0].to(device)
            logits_out=model(xb_out)
            energy_out=-temperature*torch.logsumexp(logits_out/temperature, dim=1)
            loss_energy_out=torch.pow(F.relu(m_out-energy_out), 2).mean()
            loss_energy=loss_energy_in+loss_energy_out
            total_loss=ce_loss+current_weight*loss_energy
            total_loss.backward()
            opt.step()
            ep_ce+=ce_loss.item()
            ep_energy+=loss_energy.item()
            nb+=1
            preds=logits.argmax(1)
            nc+=preds.eq(yb).sum().item()
            nt+=yb.size(0)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())
        train_loss=ep_ce/nb
        train_acc=nc/nt
        train_f1=float(f1_score(train_targets, train_preds, average="macro"))
        val_loss, val_acc, val_f1=evaluate_full(model, va_loader, device)
        stats=compute_energy_stats(model, va_loader, unk_monitor_loader, device, temperature)
        acc_drop=base_val_acc-val_acc
        eligible=(acc_drop<=max_acc_drop)
        marker=""
        if eligible and stats["gap"]>best_gap:
            best_gap=stats["gap"]
            best_state={k: v.cpu() for k,v in model.state_dict().items()}
            best_epoch=epoch+1
            torch.save(best_state, out_path)
            marker="* New best model saved"
        print(f"Ep {epoch+1:.3d}/{epochs} | weight={current_weight:.3f} ce={ep_ce/nb:.4f} energy={ep_energy/nb:.4f} | val_acc={val_acc*100:.2f}% (drop={acc_drop*100:+.2f}%) | kn={stats["kn_mean"]:.3f} unk={stats["unk_mean"]:.3f} gap={stats["gap"]:.3f} overlap={stats["overlap_at_kn_mean"]*100:.1f}%\n{marker}")
        logger.log_epoch(epoch+1, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)
    logger.finalize()
    if best_state is None:
        print(f"[WARN] No epoch respected max_acc_drop={max_acc_drop*100:.1f}%. No checkppoint energy-tuned saved, reduced energy_weight/lr or max_acc_drop in Config")
        sys.exit(1)
    print(f"Stage 0b complete. Best epoch: {best_epoch} gap={best_gap:.3f} saved to {out_path}")
    
if __name__=="__main__":
    main()