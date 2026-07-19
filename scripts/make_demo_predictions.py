#!/usr/bin/env python3
"""Build a small demo prediction file from gold answers (for sanity-checking the metric)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import format_ground_truth_answer, load_split


def main() -> None:
    rows = []
    for split in ("equation", "score", "semantic", "faithful"):
        data = load_split(split)
        for ex in data[:20]:
            gold = format_ground_truth_answer(ex.get("answer_final"))
            rows.append(
                {
                    "id": ex["id"],
                    "split": split,
                    "prediction": f"【推理过程】\n(demo)\n【最终答案】\n{gold}",
                }
            )
    out = ROOT / "outputs" / "demo_predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} demo predictions -> {out}")


if __name__ == "__main__":
    main()
