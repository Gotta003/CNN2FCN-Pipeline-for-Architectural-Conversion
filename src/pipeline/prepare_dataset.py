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

def generate_silence_samples(extracted_path: Path, samples_per_class: int, log_callback) -> list[Path]: #Generate silence samples
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
    return silence_files

def extract_features_to_cache(wav_files: list[Path], cache_path: Path, processor: AudioProcessor, feature_type: str, log_callback, safe_log) -> None:
    X_list=[]
    total_files=len(wav_files)
    cls_name=cache_path.stem
    for idx, f in enumerate(wav_files):
        y_audio, _=sf.read(str(f))
        y_int16=np.clip(y_audio*32767, -32768, 32767).astype(np.int16)
        if idx%10==0:
            progress=(idx/total_files)*100
            safe_log(f"Caching {cls_name}: [{idx}/{total_files}] {progress:.1f}% \r", replace=True)
        feat=processor.compute_features(y_int16, feature_type=feature_type)
        X_list.append(feat)
    safe_log(f"Caching {cls_name}: 100% - Saved to {cache_path.name}", replace=False)
    X_arr=np.array(X_list, dtype=np.float32)
    np.savez_compressed(cache_path, X=X_arr)
    
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
    cache_dir=root/"cache"/feature_type
    random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
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
        generate_silence_samples(extracted_path, samples_per_class, log_callback)
        
    all_dirs=[d for d in extracted_path.iterdir() if d.is_dir()]
    all_classes=[d.name for d in all_dirs if not d.name.startswith("_")]
    if "silence_generate" in [d.name for d in all_dirs]:
        all_classes.append("silence")
    else:
        generate_silence_samples(extracted_path, samples_per_class, log_callback)
        all_classes.append("silence")
    #Audio Processor with universal caching
    processor=AudioProcessor(use_gpu=use_gpu)
    log_callback(f"Audio Processor: Mode={feature_type.upper()} | device={processor.device_gpu}")
    log_callback("Checking cache for all words")
    for cls in all_classes:
        folder_name="silence_generated" if cls=="silence" else cls
        cache_file=cache_dir/f"{cls}.npz"
        if not cache_file.exists():
            wav_files=list((extracted_path/folder_name).glob("*.wav"))
            if wav_files:
                extract_features_to_cache(wav_files, cache_file, processor, feature_type, log_callback, safe_log)
    if extracted_path.exists():
        log_callback("Cleaning up extracted raw audio folder...")
        shutil.rmtree(str(extracted_path))
    #Dataset Dynamic Splits Preparation
    log_callback("Computing data splits layouts")
    target_classes=[k for k in user_keywords if k not in ("silence", "unknown")]
    target_classes.append("silence")
    target_classes=sorted(target_classes)
    label_map={cls: idx for idx, cls in enumerate(target_classes)}
    unknown_sources=sorted(list(set([c for c in all_classes if c!="silence"])-set(target_classes)))
    random.shuffle(unknown_sources)
    n_unk=len(unknown_sources)
    train_end=int(n_unk*cfg["data"]["train_unk_words"])
    train_unk_words=unknown_sources[:train_end]
    val_end=int(n_unk*(cfg["data"]["train_unk_words"]+cfg["data"]["val_unk_words"]))
    val_unk_words=unknown_sources[train_end:val_end]
    test_unk_words=unknown_sources[val_end:]
    log_callback(f"OOD Word Split Train: {len(train_unk_words)} Val: {len(val_unk_words)} Test: {len(test_unk_words)}")
    splits_data={
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []}
    }
    train_class_counts={}
    val_class_counts={}
    test_class_counts={}
    test_frac=data_cfg["test_fraction"]
    val_frac=data_cfg["val_fraction"]
    total_non_train=test_frac+val_frac
    for cls in target_classes:
        cache_file=cache_dir/f"{cls}.npz"
        X_cls=np.load(cache_file)["X"]
        y_cls=np.full(len(X_cls), label_map[cls], dtype=np.int64)
        X_train, X_temp, y_train, y_temp=train_test_split(X_cls, y_cls, test_size=total_non_train, random_state=seed)
        X_val, X_test, y_val, y_test=train_test_split(X_temp, y_temp, test_size=(test_frac/total_non_train), random_state=seed)
        splits_data["train"]["X"].append(X_train)
        splits_data["train"]["y"].append(y_train)
        splits_data["val"]["X"].append(X_val)
        splits_data["val"]["y"].append(y_val)
        splits_data["test"]["X"].append(X_test)
        splits_data["test"]["y"].append(y_test)
        train_class_counts[cls]=len(y_train)
        val_class_counts[cls]=len(y_val)
        test_class_counts[cls]=len(y_test)
        
    def append_ood(words_list, target_split_name, max_samples=None):
        if not words_list:
            return 0
        X_ood_list=[]
        for w in words_list:
            X_ood_list.append(np.load(cache_dir/f"{w}.npz")["X"])
        X_ood=np.concatenate(X_ood_list, axis=0)
        indices=np.random.permutation(len(X_ood))
        X_ood=X_ood[indices]
        if max_samples:
            X_ood=X_ood[:max_samples]
        y_ood=np.full(len(X_ood), UNKNOWN_SENTINEL, dtype=np.int64)
        splits_data[target_split_name]["X"].append(X_ood)
        splits_data[target_split_name]["y"].append(y_ood)
        return len(X_ood)
    
    train_unk_count=append_ood(train_unk_words, "train", max_samples=samples_per_class)
    val_unk_count=append_ood(val_unk_words, "val", max_samples=samples_per_class//2)
    test_unk_count=append_ood(test_unk_words, "test", max_samples=samples_per_class//2)
    for split_name, data in splits_data.items():
        X_arr=np.concatenate(data["X"], axis=0)
        y_arr=np.concatenate(data["y"], axis=0)
        np.savez_compressed(out_dir/f"dataset_{split_name}.npz", X=X_arr, y=y_arr)
        log_callback(f"Saved {split_name} split:  {len(X_arr)} samples")
    
    n_classes=len(target_classes)
    manifest={
        "source_path": str(archive_path.resolve()),
        "source_hash": _file_hash(archive_path),
        "seed": seed,
        "test_fraction": test_frac,
        "val_fraction": val_frac,
        "unknown_sentinel": UNKNOWN_SENTINEL,
        "class_names": target_classes,
        "class_names_all": target_classes+["unknown"],
        "num_classes": n_classes,
        "train_samples": sum(len(f) for f in splits_data["train"]["y"]),
        "train_unknown_samples": train_unk_count,
        "train_class_counts": train_class_counts,
        "val_samples": sum(len(f) for f in splits_data["val"]["y"]),
        "val_unknown_samples": val_unk_count,
        "val_class_counts": val_class_counts,
        "test_samples": sum(len(f) for f in splits_data["test"]["y"]),
        "test_unknown_samples": test_unk_count,
        "test_class_counts": test_class_counts,
        "feature_type": feature_type.upper(),
    }
    with open(out_dir/"dataset_manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    shutil.copy(out_dir/"dataset_manifest.json", out_dir/"manifest.json")
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
    print(f"Train: {manifest['train_samples']:,} (+{manifest['train_unknown_samples']} unknown)")
    print(f"Val full: {manifest['val_samples']:,} (+{manifest['val_unknown_samples']} unknown)")
    print(f"Test: {manifest['test_samples']:,} (+{manifest['test_unknown_samples']} unknown)")
    print(f"num_classes (model output): {manifest['num_classes']}")
    
if __name__=="__main__":
    main()