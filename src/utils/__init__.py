import json
import hashlib
import yaml
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_metadata(output_dir: Path, **kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def dataset_hash(data: list) -> str:
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
