from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path

UNKNOWN_SENTINEL=-1

class DatasetNotPreparedError(RuntimeError):
    pass

REQUIRED_FILES=["dataset_train.npz", "dataset_val_known.npz", "dataset_val_full.npz", "dataset_test.npz"]

def is_dataset_prepared(root: Path) -> bool:
    return all((root/"data"/f).exists() for f in REQUIRED_FILES)

def load_manifest(root: Path) -> dict | None:
    p=root/"data"/"dataset_manifest.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def extract_audio_features(X_data: np.ndarray) -> np.ndarray:
        B=X_data.shape[0]
        X_=X_data
        F_, T=X_.shape[1], X_.shape[2]
        eps=1e-8
        feats=[]
        freq_ax=np.arange(F_)
        band_pow=X_.sum(axis=2)+eps
        total_band=band_pow.sum(axis=1, keepdims=True)+eps
        
        sc=(band_pow*freq_ax[None, :]).sum(1)/total_band.squeeze()
        sv=((freq_ax[None, :]-sc[:, None])**2 * band_pow).sum(1)/total_band.squeeze()
        feats+=[sc[:, None], np.sqrt(sv)[:, None]]
        cum=np.cumsum(band_pow, axis=1)
        feats.append(np.argmax(cum>=0.85*total_band, axis=1)[:, None])
        for lo, hi in [(0, F_//3), (F_//3, 2*F_//3), (2*F_//3, F_)]:
            feats.append(X_[:, lo:hi, :].mean(axis=(1, 2))[:, None])
        frame_pow=X_.sum(axis=1)+eps
        tc=(frame_pow*np.arange(T)[None, :]).sum(1)/(frame_pow.sum(1)+eps)
        feats.append(tc[:, None])
        
        fd=np.diff(X_, axis=1)
        feats.append((fd[:, :-1, :]*fd[:, 1:, :]<0).mean(axis=(1, 2))[:, None])
        for b in range(13):
            s, e=int(b*F_/13), int((b+1)*F_/13)
            feats.append(X_[:, s:e, :].mean(axis=(1, 2))[:, None])
        feats.append(np.maximum(0, np.diff(frame_pow, axis=1)).mean(1)[:, None])
        feats.append(frame_pow.mean(1)[:, None])
        feats.append(frame_pow.std(1)[:, None])
        fv=X_.var(axis=1).mean(1)
        tv=X_.var(axis=2).mean(1)
        feats.append((tv/(fv+eps))[:, None])
        return np.concatenate(feats, axis=1).astype(np.float32)

def to_full_flat(X_2d: np.ndarray) -> np.ndarray:
        tr=extract_audio_features(X_2d)
        return np.concatenate([X_2d.reshape(len(X_2d), -1), tr], axis=1)

def _to_spatial(arr: np.ndarray) -> torch.Tensor:
    return torch.FloatTensor(arr).unsqueeze(1)

def load_splits(cfg: dict, root: Path):
    data_cfg=cfg["data"]
    missing=[f for f in REQUIRED_FILES if not (root/"data"/f).exists()]
    if missing:
        raise DatasetNotPreparedError("Dataset has not been prepared yet. Missing: "+", ".join(missing)+"\nRun dataset preparation first: the Data panel's 'Prepare dataset' button")
    tr=np.load(root/"data"/"dataset_train.npz")
    va_k=np.load(root/"data"/"dataset_val_known.npz")
    va_full=np.load(root/"data"/"dataset_val_full.npz")
    te=np.load(root/"data"/"dataset_test.npz")
    X_tr, y_tr=tr["X"], tr["y"].astype(np.int64)
    X_vak, y_vak=va_k["X"], va_k["y"].astype(np.int64)
    X_vaf, y_vaf=va_full["X"], va_full["y"].astype(np.int64)
    X_te, y_te=te["X"], te["y"].astype(np.int64)
    n_classes=data_cfg["num_classes"]
    class_names=data_cfg["class_names"]
    
    X_tr_flat=to_full_flat(X_tr)
    X_vak_flat=to_full_flat(X_vak)
    X_vaf_flat=to_full_flat(X_vaf)
    X_te_flat=to_full_flat(X_te)
    #Class weights
    counts=np.bincount(y_tr, minlength=n_classes).clip(1).astype(float)
    w=len(y_tr)/(n_classes*counts)
    w=w/w.sum()*n_classes
    class_weights=torch.FloatTensor(w)
    cw=class_weights.clone()
    if "silence" in class_names:
        silence_idx=class_names.index("silence")
        cw[silence_idx]=cw[silence_idx]*0.7
    cw=(cw/cw.sum()*n_classes)

    return {
        "X_train_spatial": _to_spatial(X_tr),
        "X_train_full": torch.FloatTensor(X_tr_flat),
        "y_train": torch.LongTensor(y_tr),
        "X_val_known_spatial": _to_spatial(X_vak),
        "X_val_known_full": torch.FloatTensor(X_vak_flat),
        "y_val_known": torch.LongTensor(y_vak),
        "X_val_full_spatial": _to_spatial(X_vaf),
        "X_val_full_flat": torch.FloatTensor(X_vaf_flat),
        "y_val_full": torch.LongTensor(y_vaf),
        "X_test_spatial": _to_spatial(X_te),
        "X_test_full": torch.FloatTensor(X_te_flat),
        "y_test": torch.LongTensor(y_te),
        "n_classes": n_classes,
        "known_indices": list(range(n_classes)),
        "unknown_sentinel": UNKNOWN_SENTINEL,
        "class_names": class_names,
        "class_weights": class_weights,
        "cw": cw, 
    }
    
