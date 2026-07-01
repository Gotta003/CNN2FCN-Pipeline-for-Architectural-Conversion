from __future__ import annotations
import json
import random
import shutil
import argparse
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
import sys
project_root=str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)
from src.pipeline.audio_processor import AudioProcessor

UNKNOWN_SENTINEL=-1

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--features", default="mfe", choices=["mfe", "mfcc"])
    p.add_argument("--cpu-only", action="store_true")
    return p.parse_args()

def _file_hash(path: Path, chunk_size: int=1<<20)-> str:
    if not path.exists():
        return "N/A"
    h=hashlib.sha256()
    with open(path, "rb") as f:
        while chunk:=f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]

def pipeline_dataset_builder(root: str | Path, user_keywords: list[str], cfg: dict, feature_type: str="mfe", use_gpu: bool=True, samples_per_class: int=3800, log_callback=print):
    def safe_log(msg, replace=False):
        try:
            log_callback(msg, replace=replace)
        except TypeError:
            log_callback(msg.replace("\r", ""))
    
    data_cfg=cfg["data"]
    seed=data_cfg["seed"]
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    archive_path=Path("speech_commands_v0.02.tar.gz")
    extracted_path=Path("speech_commands")
    root=Path(root).resolve()
    out_dir=root/"data"
    random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    #Download 
    if not archive_path.exists():
        log_callback(f"Downloading dataset from {url}")
        import requests
        res=requests.get(url, stream=True)
        with open(archive_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    #Extract WAV
    if not extracted_path.exists():
        log_callback("Extract archive tracks")
        shutil.unpack_archive(str(archive_path), str(extracted_path))
    target_classes=[k for k in user_keywords if k not in ("silence", "unknown")]
    target_classes.append("silence")
    target_classes=sorted(target_classes)
    all_dirs=[d for d in extracted_path.iterdir() if d.is_dir()]
    all_classes=[d.name for d in all_dirs if not d.name.startswith("_")]
    unknown_sources=sorted(set(all_classes)-set(target_classes))
    class_to_files={cls: list((extracted_path/cls).glob("*.wav")) for cls in target_classes if cls!="silence"}
    
    unk_pool=[]
    for cls in unknown_sources:
        unk_pool.extend(list((extracted_path/cls).glob("*.wav")))
    random.shuffle(unk_pool)
    class_to_files["unknown"]=unk_pool[:samples_per_class]
    #Generate silence samples
    log_callback("Generating silence samples")
    silence_generated=extracted_path/"silence_generated"
    silence_generated.mkdir(exist_ok=True)
    bg_noise_dir=extracted_path/"_background_noise_"
    
    all_bg_audio=[]
    for f in bg_noise_dir.glob("*.wav"):
        y, _=sf.read(str(f))
        if len(y.shape)>1:
            y=y[:, 0]
        all_bg_audio.append(y)
    
    long_noise=np.concatenate(all_bg_audio, axis=0)
    silence_files=[]
    for i in range(min(samples_per_class, 3800)):
        start=random.randint(0, len(long_noise)-16000)
        clip=long_noise[start:start+16000]
        f_out=silence_generated/f"silence_{i:04d}.wav"
        sf.write(str(f_out), clip, 16000)
        silence_files.append(f_out)
    class_to_files["silence"]=silence_files
    #Audio Processor
    processor=AudioProcessor(use_gpu=use_gpu)
    log_callback(f"Audio Processor: Mode={feature_type.upper()} | device={processor.device_gpu}")
    #Dataset Partition
    log_callback("Computing data splits layouts")
    splits_files={"train": {}, "val_known": {}, "val_full": {}, "test": {}}
    test_frac=data_cfg["test_fraction"]
    val_frac=data_cfg["val_fraction"]
    for idx, (cls, files) in enumerate(class_to_files.items()):
        if cls=="unknown":
            val, test=train_test_split(files, test_size=0.5, random_state=seed)
            splits_files["val_full"]["unknown"]=val
            splits_files["test"]["unknown"]=test
        else:
            total_non_train=test_frac+val_frac
            train, temp=train_test_split(files, test_size=total_non_train, random_state=seed)
            val, test=train_test_split(temp, test_size=(test_frac/total_non_train), random_state=seed)
            splits_files["train"][cls]=train
            splits_files["val_known"][cls]=val
            splits_files["val_full"][cls]=val
            splits_files["test"][cls]=test
        log_callback(f"Completed split class {cls} [{idx}]")

    label_map={cls: idx for idx, cls in enumerate(target_classes)}
    if "silence" not in label_map:
        label_map["silence"]=len(label_map)
        target_classes.append("silence")
    for split_name, class_dict in splits_files.items():
        X_list, y_list=[], []
        for cls, files in class_dict.items():
            total_files=len(files)
            lbl_id=UNKNOWN_SENTINEL if cls=="unknown" else label_map[cls]
            for idx, f in enumerate(files):
                y_audio, _=sf.read(str(f))
                y_int16=np.clip(y_audio*32767, -32768, 32767).astype(np.int16)
                if idx%10==0:
                    progress=(idx/total_files)*100
                    safe_log(f"Extracting {cls}: [{idx}/{total_files}] {progress:.1f}% \r", replace=True)
                feat=processor.compute_features(y_int16, feature_type=feature_type)
                X_list.append(feat)
                y_list.append(lbl_id)
            safe_log(f"Extracting {cls}: 100% - Complete!", replace=False)
        X_arr=np.array(X_list, dtype=np.float32)
        y_arr=np.array(y_list, dtype=np.int64)
        np.savez_compressed(out_dir/f"dataset_{split_name}.npz", X=X_arr, y=y_arr)
    
    def class_counts(split_dict, include_unknown=False):
        counts={cls: len(split_dict.get(cls, [])) for cls in target_classes}
        if include_unknown:
            counts["unknown"]=len(split_dict.get("unknown", []))
        return counts
    
    n_classes=len(target_classes)
    known_files_count=sum(len(f) for c, f in class_to_files.items() if c!="unknown")
    unk_files_count=len(class_to_files.get("unknown", []))
    manifest={
        "source_path": str(archive_path.resolve()),
        "source_hash": _file_hash(archive_path),
        "seed": seed,
        "test_fraction": test_frac,
        "val_fraction": val_frac,
        "unknown_sentinel": UNKNOWN_SENTINEL,
        "class_names": target_classes,
        "class_names_all": target_classes+["unknown"],
        "unknown_label": "unknown",
        "num_classes": n_classes,
        "total_samples": unk_files_count,
        "known_samples": known_files_count, 
        "unknown_samples": unk_files_count,
        "train_samples": sum(len(f) for f in splits_files["train"].values()),
        "val_known_samples": sum(len(f) for f in splits_files["val_known"].values()),
        "val_unknown_samples": len(splits_files["val_full"].get("unknown", [])),
        "val_full_samples": sum(len(f) for f in splits_files["val_known"].values())+len(splits_files["val_full"].get("unknown", [])), 
        "test_samples": sum(len(f) for f in splits_files["test"].values()),
        "test_unknown_samples": len(splits_files["test"].get("unknown", [])),
        "train_class_counts": class_counts(splits_files["train"]),
        "val_known_class_counts": class_counts(splits_files["val_known"]),
        "val_full_class_counts": class_counts(splits_files["val_full"], include_unknown=True),
        "test_class_counts": class_counts(splits_files["test"], include_unknown=True),
        "feature_type": feature_type.upper(),
    }
    with open(out_dir/"dataset_manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    shutil.copy(out_dir/"dataset_manifest.json", out_dir/"manifest.json")
    if extracted_path.exists():
        shutil.rmtree(str(extracted_path))
    log_callback("Dataset preparation completed")
    return manifest

def main():
    args=parse_args()
    root=Path(args.root).resolve()
    import yaml
    with open(args.config) as f:
        cfg=yaml.safe_load(f)
    print(f"{'='*60}\nOPEN-SET DATASET PREPARATION\n{'='*60}")
    manifest=pipeline_dataset_builder(root=root, user_keywords=cfg["data"]["class_names"], cfg=cfg, feature_type=args.features, use_gpu=not args.cpu_only, samples_per_class=3800)
    print(f"\nDone. Source hash: {manifest['source_hash']}")
    print(f"Train: {manifest['train_samples']:,} (known only)")
    print(f"Val known: {manifest['val_known_samples']:,}")
    print(f"Val full: {manifest['val_full_samples']:,} (+{manifest['val_unknown_samples']} unknown)")
    print(f"Test: {manifest['test_samples']:,} (+{manifest['test_unknown_samples']} unknown)")
    print(f"num_classes (model output): {manifest['num_classes']}")
    
if __name__=="__main__":
    main()