from __future__ import annotations

import argparse
import os
from pathlib import Path


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DATASET_ID = "databricks/databricks-dolly-15k"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache real Qwen benchmark assets")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    from datasets import DownloadConfig, load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    snapshot = Path(
        snapshot_download(MODEL_ID, local_files_only=args.local_files_only)
    )
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise RuntimeError(f"unexpected model type: {config.model_type}")
    weight_files = tuple(snapshot.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError(f"no safetensors weights found in {snapshot}")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        download_mode="reuse_dataset_if_exists",
        download_config=DownloadConfig(local_files_only=args.local_files_only),
    )
    required_rows = 4096 + 512
    if len(dataset) < required_rows:
        raise RuntimeError(f"Dolly has {len(dataset)} rows; need {required_rows}")
    required_columns = {"instruction", "context", "response"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"Dolly is missing columns: {sorted(missing)}")

    size = sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
    print(f"model_snapshot={snapshot}")
    print(f"model_bytes={size}")
    print(f"weight_files={len(weight_files)}")
    print(f"dataset_rows={len(dataset)}")
    print(f"hf_home={os.environ.get('HF_HOME', '(default)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
