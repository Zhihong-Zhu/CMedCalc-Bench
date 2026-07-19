"""Dataset loading and field helpers for CMedCalc-Bench."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SPLIT_FILES = {
    "equation": "equation.json",
    "score": "score.json",  # rule-based accumulators
    "semantic": "semantic.json",
    "faithful": "faithful.json",
}

# Friendly aliases used in the paper
SPLIT_ALIASES = {
    "equation": "equation",
    "equation-based": "equation",
    "rule": "score",
    "rule-based": "score",
    "score": "score",
    "semantic": "semantic",
    "semantic-based": "semantic",
    "faithful": "faithful",
}


def resolve_split(name: str) -> str:
    key = name.strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(
            f"Unknown split '{name}'. Choose from: {sorted(set(SPLIT_ALIASES))}"
        )
    return SPLIT_ALIASES[key]


def load_split(split: str, data_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    split = resolve_split(split)
    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / SPLIT_FILES[split]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def load_all(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[Dict[str, Any]]]:
    return {name: load_split(name, data_dir=data_dir) for name in SPLIT_FILES}


def get_patient_note(example: Dict[str, Any]) -> str:
    note = example.get("record_final", "")
    if isinstance(note, (dict, list)):
        return json.dumps(note, ensure_ascii=False, indent=2)
    return str(note)


def get_calculator_name(example: Dict[str, Any]) -> str:
    return str(example.get("指标名称", example.get("calculator", "")))


def get_department(example: Dict[str, Any]) -> str:
    return str(example.get("科室", ""))


def format_ground_truth_answer(answer: Any) -> str:
    """Serialize ground-truth answers into a compact printable string."""
    if isinstance(answer, dict):
        # Nested multi-output (e.g., supine/sitting PaO2)
        if all(isinstance(v, dict) and "value" in v for v in answer.values()):
            parts = []
            for k, v in answer.items():
                unit = v.get("unit", "")
                parts.append(f"{k}: {v.get('value')}{' ' + unit if unit else ''}".rstrip())
            return "; ".join(parts)
        if "value" in answer:
            unit = answer.get("unit", "")
            value = answer.get("value")
            return f"{value}{' ' + unit if unit else ''}".rstrip()
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def extract_numeric_values(text: str) -> List[float]:
    """Extract floats/ints from free-form model output."""
    if text is None:
        return []
    text = str(text)
    # Normalize Chinese punctuation / full-width digits lightly
    text = text.replace("，", ",").replace("．", ".")
    pattern = r"[-+]?(?:\d+\.\d+|\d+)"
    vals = []
    for m in re.finditer(pattern, text):
        try:
            vals.append(float(m.group()))
        except ValueError:
            continue
    return vals


def is_date_like_unit(unit: str, calculator_name: str = "") -> bool:
    unit = unit or ""
    name = calculator_name or ""
    date_keywords = ("日期", "孕周", "周", "天")
    name_keywords = ("预产期", "受孕日", "估计孕周", "日期")
    return any(k in unit for k in date_keywords) or any(k in name for k in name_keywords)


def iter_examples(
    splits: Optional[Iterable[str]] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> Iterable[Dict[str, Any]]:
    names = list(splits) if splits else list(SPLIT_FILES)
    for name in names:
        for ex in load_split(name, data_dir=data_dir):
            item = dict(ex)
            item["_split"] = resolve_split(name)
            yield item
