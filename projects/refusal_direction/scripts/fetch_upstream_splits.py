from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9d852fae1a9121c78b29142de733cb1340770cc3"
BASE = (
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/"
    f"{COMMIT}/dataset/splits"
)
SPLITS = (
    "harmful_train",
    "harmful_val",
    "harmless_train",
    "harmless_val",
    "harmful_test",
    "harmless_test",
)


def main() -> int:
    destination = ROOT / "artifacts" / "datasets"
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with urlopen(f"{BASE}/{split}.json", timeout=60) as response:
            rows = json.load(response)
        label = "harmful" if split.startswith("harmful") else "harmless"
        normalized = [
            {
                "id": f"{split}-{index:04d}",
                "text": row["instruction"],
                "label": label,
                "source": f"upstream:{COMMIT}:{split}",
            }
            for index, row in enumerate(rows)
        ]
        (destination / f"{split}.json").write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
