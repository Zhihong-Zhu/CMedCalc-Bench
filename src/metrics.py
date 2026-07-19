"""Answer matching metrics for CMedCalc-Bench.

Following the paper:
- Equation-based (non-date): ±5% relative tolerance on numerical values
- Equation-based (date-related): exact match
- Rule-based / semantic-based: strict exact match
- Faithful reasoning: success if the model abstains (refuses to compute)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .data_utils import extract_numeric_values, format_ground_truth_answer, is_date_like_unit

REFUSAL_PATTERNS = [
    r"不能计算",
    r"无法计算",
    r"无法准确计算",
    r"无法评估",
    r"无法得出",
    r"信息不足",
    r"参数缺失",
    r"无法应用",
    r"cannot\s+compute",
    r"unable\s+to\s+calculate",
    r"insufficient\s+information",
    r"not\s+enough\s+information",
]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.replace(" ", "").replace("\n", "").replace("\t", "")
    s = s.replace("：", ":").replace("（", "(").replace("）", ")")
    s = s.replace("％", "%").replace("㎎", "mg")
    # strip common wrappers
    s = re.sub(r"^[\"'`]+|[\"'`]+$", "", s)
    return s


def is_refusal(prediction: Any) -> bool:
    text = str(prediction or "")
    return any(re.search(p, text, flags=re.IGNORECASE) for p in REFUSAL_PATTERNS)


def _within_tolerance(pred: float, gold: float, tol: float = 0.05) -> bool:
    if gold == 0:
        return abs(pred - gold) <= 1e-6
    return abs(pred - gold) / abs(gold) <= tol


def _extract_gold_numeric_targets(answer: Any) -> List[Tuple[float, str]]:
    """Return list of (value, unit) from structured gold answers."""
    targets: List[Tuple[float, str]] = []

    def add_value(value: Any, unit: str = "") -> None:
        nums = extract_numeric_values(str(value))
        if nums:
            targets.append((nums[0], unit or ""))

    if isinstance(answer, dict):
        if "value" in answer and not any(isinstance(v, dict) and "value" in v for v in answer.values()):
            add_value(answer.get("value"), str(answer.get("unit", "")))
        else:
            for v in answer.values():
                if isinstance(v, dict) and "value" in v:
                    add_value(v.get("value"), str(v.get("unit", "")))
                else:
                    add_value(v, "")
    else:
        add_value(answer, "")
    return targets


def match_equation(
    prediction: Any,
    gold_answer: Any,
    calculator_name: str = "",
    tol: float = 0.05,
) -> bool:
    pred_text = str(prediction or "")
    gold_text = format_ground_truth_answer(gold_answer)

    # Date / gestational-age style: require exact normalized match or exact date string presence
    unit = ""
    if isinstance(gold_answer, dict) and "unit" in gold_answer:
        unit = str(gold_answer.get("unit") or "")
    if is_date_like_unit(unit, calculator_name):
        gold_norm = normalize_text(gold_text)
        pred_norm = normalize_text(pred_text)
        if gold_norm and gold_norm in pred_norm:
            return True
        # also try raw value field
        if isinstance(gold_answer, dict) and "value" in gold_answer:
            gv = normalize_text(gold_answer.get("value"))
            if gv and gv in pred_norm:
                return True
        return pred_norm == gold_norm

    targets = _extract_gold_numeric_targets(gold_answer)
    if not targets:
        return normalize_text(pred_text) == normalize_text(gold_text)

    pred_nums = extract_numeric_values(pred_text)
    if not pred_nums:
        return False

    # Multi-output: every gold number must match some predicted number within tolerance
    if len(targets) > 1:
        used = set()
        for gval, _ in targets:
            found = False
            for i, pval in enumerate(pred_nums):
                if i in used:
                    continue
                if _within_tolerance(pval, gval, tol=tol):
                    used.add(i)
                    found = True
                    break
            if not found:
                return False
        return True

    gval = targets[0][0]
    # Prefer the last number (often the final answer), but accept any in-tolerance hit
    if _within_tolerance(pred_nums[-1], gval, tol=tol):
        return True
    return any(_within_tolerance(p, gval, tol=tol) for p in pred_nums)


def match_exact(prediction: Any, gold_answer: Any) -> bool:
    pred = normalize_text(prediction)
    gold = normalize_text(format_ground_truth_answer(gold_answer))
    if not gold:
        return False
    if pred == gold:
        return True
    # Allow gold answer as a standalone token inside prediction
    # e.g., prediction ends with "最终答案：3级"
    if re.search(rf"(^|[^0-9a-z\u4e00-\u9fff]){re.escape(gold)}([^0-9a-z\u4e00-\u9fff]|$)", pred):
        return True
    return False


def match_faithful(prediction: Any, gold_answer: Any = None) -> bool:
    """Faithful split: correct if the model refuses to compute."""
    if is_refusal(prediction):
        return True
    # Some gold labels are explicit refusal strings; also accept exact match
    if gold_answer is not None and is_refusal(gold_answer):
        return is_refusal(prediction)
    return False


def evaluate_example(
    prediction: Any,
    example: Dict[str, Any],
    split: Optional[str] = None,
    tol: float = 0.05,
) -> Dict[str, Any]:
    split = (split or example.get("_split") or "").lower()
    gold = example.get("answer_final")
    calc = str(example.get("指标名称", ""))

    if split in {"faithful"}:
        correct = match_faithful(prediction, gold)
        criterion = "refusal"
    elif split in {"equation", "equation-based"}:
        correct = match_equation(prediction, gold, calculator_name=calc, tol=tol)
        criterion = "numeric±5%" if not is_date_like_unit(
            str(gold.get("unit", "")) if isinstance(gold, dict) else "", calc
        ) else "exact-date"
    else:
        # score / semantic / rule-based
        correct = match_exact(prediction, gold)
        criterion = "exact"

    return {
        "correct": bool(correct),
        "criterion": criterion,
        "prediction": prediction,
        "gold": gold,
        "gold_str": format_ground_truth_answer(gold),
        "calculator": calc,
        "split": split,
        "id": example.get("id"),
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"accuracy": 0.0, "n": 0, "n_correct": 0, "by_split": {}, "by_calculator": {}}

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    by_split: Dict[str, Dict[str, Union[int, float]]] = {}
    by_calc: Dict[str, Dict[str, Union[int, float]]] = {}

    for r in results:
        for key, bucket in (("split", by_split), ("calculator", by_calc)):
            name = str(r.get(key) or "unknown")
            if name not in bucket:
                bucket[name] = {"n": 0, "n_correct": 0, "accuracy": 0.0}
            bucket[name]["n"] += 1
            bucket[name]["n_correct"] += int(bool(r.get("correct")))

    for bucket in (by_split, by_calc):
        for v in bucket.values():
            v["accuracy"] = round(v["n_correct"] / v["n"], 4) if v["n"] else 0.0

    return {
        "accuracy": round(n_correct / n, 4),
        "n": n,
        "n_correct": n_correct,
        "by_split": by_split,
        "by_calculator": by_calc,
    }


def dump_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
