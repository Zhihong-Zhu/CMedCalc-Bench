"""Prompt builders for CMedCalc-Bench (Direct / Zero-shot CoT / One-shot CoT)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .data_utils import format_ground_truth_answer, get_calculator_name, get_patient_note

# Paper-style one-shot exemplars (short, human-readable)
DEFAULT_EXEMPLARS = {
    "equation": {
        "指标名称": "体重指数(BMI)",
        "record_final": (
            "主诉:\n\n27岁女性\n\n现病史:\n\n婚后1年性生活正常，未避孕。"
            "\n\n查体:\n\n婚前体重105斤，现体重130斤，身高162厘米。"
        ),
        "explanation_final": (
            "计算公式：体重指数（BMI）=体重/身高^2\n"
            "根据病历，患者当前体重为130斤（等于65kg），身高为162厘米（等于1.62m）。\n"
            "BMI = 65 / (1.62 * 1.62) = 65 / 2.6244 ≈ 24.8"
        ),
        "answer_final": {"value": "24.8", "unit": "kg/m²"},
    },
    "score": {
        "指标名称": "大出血风险HAS-BLED评分",
        "record_final": (
            "22岁男性，高血压病史，确诊肾脏疾病与肝硬化体征，"
            "有短暂性脑缺血发作史，每周饮酒12单位，无出血史，INR稳定，未使用增加出血风险药物。"
        ),
        "explanation_final": (
            "当前HAS-BLED评分为0分。患者年龄22岁（未达65岁不评分）。"
            "存在高血压病史（H）加1分；肾脏疾病（A）加1分；肝硬化（A）加1分；"
            "卒中/TIA史（S）加1分；饮酒≥8单位/周（D）加1分。"
            "最终HAS-BLED总评分为5分。"
        ),
        "answer_final": "5",
    },
    "semantic": {
        "指标名称": "Levine 6级分级法（心脏杂音）",
        "record_final": (
            "1岁3月女婴，精神差、间断呻吟3天伴呕吐、稀便。"
            "查体：呼吸60次/分，心率150-160次/分，心肺听诊：双肺呼吸音粗，心音有力，未闻及杂音。"
        ),
        "explanation_final": (
            "0级：查体部分明确记录“未闻及杂音”。尽管患儿存在心动过速等体征，"
            "但心脏听诊未发现杂音，因此评为0级。"
        ),
        "answer_final": "0级",
    },
    "faithful": {
        "指标名称": "烧伤面积(九分法)",
        "record_final": (
            "手背：大面积皮肤热轧伤，水泡，部分表皮腊白。手掌：皮肤看着还行。"
            "第二次：手掌皮肤看着还可以，还没有坏死。"
        ),
        "explanation_final": (
            "病历中仅提及手背和手掌的烧伤，并使用了“大面积”这一模糊描述，"
            "未提供可用于九分法计算的具体部位面积比例，因此不能计算。"
        ),
        "answer_final": "不能计算",
    },
}


DIRECT_TEMPLATE = """你是一名严谨的临床计算助手。请根据患者病历，完成指定的医学计算任务。
只输出最终答案，不要输出推理过程。

【计算任务】
{calculator}

【患者病历】
{note}

【最终答案】"""


ZERO_SHOT_COT_TEMPLATE = """你是一名严谨的临床计算助手。请根据患者病历，完成指定的医学计算任务。
请先逐步推理，再给出最终答案。若信息不足或存在矛盾导致无法计算，请明确回答“不能计算”。

【计算任务】
{calculator}

【患者病历】
{note}

请按如下格式输出：
【推理过程】
...
【最终答案】
..."""


ONE_SHOT_COT_TEMPLATE = """你是一名严谨的临床计算助手。请根据患者病历，完成指定的医学计算任务。
下面先给出一个示例，请仿照示例的推理风格作答。若信息不足或存在矛盾导致无法计算，请明确回答“不能计算”。

【示例】
计算任务：{ex_calculator}
患者病历：
{ex_note}
推理过程：
{ex_explanation}
最终答案：{ex_answer}

【正式问题】
计算任务：{calculator}
患者病历：
{note}

请按如下格式输出：
【推理过程】
...
【最终答案】
..."""


UNANSWERABLE_ONE_SHOT_TEMPLATE = """你是一名严谨的临床计算助手。请根据患者病历，完成指定的医学计算任务。
下面先给出一个“无法计算”的示例。当病历缺少关键参数或信息矛盾时，你应拒绝给出数值答案。

【示例】
计算任务：{ex_calculator}
患者病历：
{ex_note}
推理过程：
{ex_explanation}
最终答案：{ex_answer}

【正式问题】
计算任务：{calculator}
患者病历：
{note}

请按如下格式输出：
【推理过程】
...
【最终答案】
..."""


def _fill_example_fields(example: Dict[str, Any]) -> Dict[str, str]:
    return {
        "calculator": get_calculator_name(example),
        "note": get_patient_note(example),
        "explanation": str(example.get("explanation_final", "")),
        "answer": format_ground_truth_answer(example.get("answer_final")),
    }


def build_prompt(
    example: Dict[str, Any],
    style: str = "zero_shot_cot",
    split: Optional[str] = None,
    exemplar: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a prompt for a single instance.

    Args:
        example: dataset instance
        style: one of {direct, zero_shot_cot, one_shot_cot, one_shot_unanswerable}
        split: equation / score / semantic / faithful
        exemplar: optional custom one-shot exemplar
    """
    style = style.lower().replace("-", "_")
    split = (split or example.get("_split") or "equation").lower()
    if split in {"rule", "rule-based"}:
        split = "score"
    if split in {"equation-based"}:
        split = "equation"
    if split in {"semantic-based"}:
        split = "semantic"

    fields = _fill_example_fields(example)

    if style in {"direct", "direct_answer", "d"}:
        return DIRECT_TEMPLATE.format(**fields)

    if style in {"zero_shot_cot", "zero_shot", "zc", "cot"}:
        return ZERO_SHOT_COT_TEMPLATE.format(**fields)

    if style in {"one_shot_cot", "one_shot", "oc"}:
        ex = exemplar or DEFAULT_EXEMPLARS.get(split) or DEFAULT_EXEMPLARS["equation"]
        ex_fields = _fill_example_fields(ex)
        return ONE_SHOT_COT_TEMPLATE.format(
            ex_calculator=ex_fields["calculator"],
            ex_note=ex_fields["note"],
            ex_explanation=ex_fields["explanation"],
            ex_answer=ex_fields["answer"],
            calculator=fields["calculator"],
            note=fields["note"],
        )

    if style in {"one_shot_unanswerable", "unanswerable", "faithful_one_shot"}:
        ex = exemplar or DEFAULT_EXEMPLARS["faithful"]
        ex_fields = _fill_example_fields(ex)
        return UNANSWERABLE_ONE_SHOT_TEMPLATE.format(
            ex_calculator=ex_fields["calculator"],
            ex_note=ex_fields["note"],
            ex_explanation=ex_fields["explanation"],
            ex_answer=ex_fields["answer"],
            calculator=fields["calculator"],
            note=fields["note"],
        )

    raise ValueError(
        f"Unknown prompt style '{style}'. "
        "Choose from: direct, zero_shot_cot, one_shot_cot, one_shot_unanswerable"
    )


def extract_final_answer(model_output: str) -> str:
    """Heuristic extraction of the final answer span from model output."""
    if not model_output:
        return ""
    text = model_output.strip()
    markers = ["【最终答案】", "最终答案", "Final Answer", "Answer:"]
    for marker in markers:
        if marker in text:
            part = text.split(marker)[-1].strip()
            # cut trailing sections if any
            for stop in ["【推理过程】", "\n\n【"]:
                if stop in part:
                    part = part.split(stop)[0].strip()
            return part.strip(" ：:-\n")
    # fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text
