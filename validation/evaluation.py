"""Conservative, versioned grading for RPT's short-answer development suite.

This is deliberately an exact-match evaluator, not a semantic judge. Additional
wording is rejected unless explicitly listed as an accepted answer. In particular,
an answer containing a correct word followed by a contradiction is not accepted.
The historical 100-question suite was used for recovery training; its scores are
development scores even when the same examples run through a new backend.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Any, Iterable, Mapping


EVALUATOR_VERSION = "rpt-exact-match-v2"
MATCH_METHODS = frozenset({"exact", "numeric", "code"})
EVALUATION_CLASSIFICATIONS = frozenset({"development", "external_validation", "held_out"})


def _unwrap_markdown(text: str) -> str:
    """Remove answer presentation, while preserving punctuation inside code."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:[a-zA-Z0-9_+.-]+)?\s*\n(.*?)\n?```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("`") and text.endswith("`") and "\n" not in text:
        text = text[1:-1].strip()
    # Some historical generations ended after an opening bold marker. Removing
    # those wrappers does not remove any actual answer token.
    text = re.sub(r"^(?:\*\*|__)+", "", text)
    text = re.sub(r"(?:\*\*|__)+$", "", text)
    return text.strip()


def normalize_text(text: Any) -> str:
    """Normalize formatting for full-answer equality, never substring matching."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _unwrap_markdown(value)
    value = value.strip(' \t\r\n"“”‘’')
    value = value.rstrip(".!?").strip()
    value = _unwrap_markdown(value)
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()


def first_sentence(text: Any) -> str:
    """Legacy utility only. Grading intentionally checks the entire answer."""
    value = "" if text is None else str(text).strip()
    return re.split(r"(?<=[.!?])\s+", value, maxsplit=1)[0]


def _number(text: Any) -> Decimal | None:
    value = normalize_text(text)
    # No implicit thousands/decimal-comma conversion: that would make 1,000
    # ambiguous. Such representations need explicit accepted aliases.
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?", value):
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def answer_is_correct(
    prediction: Any,
    accepted_answers: Iterable[Any],
    *,
    method: str = "exact",
) -> bool:
    """Grade a complete answer using one explicitly selected comparison method."""
    if method not in MATCH_METHODS:
        raise ValueError(f"Unsupported matching method: {method!r}")
    if isinstance(accepted_answers, (str, bytes)):
        raise TypeError("accepted_answers must be a list of complete answers")
    accepted = list(accepted_answers)
    if method == "code":
        pred = _unwrap_markdown("" if prediction is None else str(prediction))
        return bool(pred) and any(
            pred == _unwrap_markdown(str(answer))
            for answer in accepted
            if answer is not None and str(answer).strip()
        )
    pred = normalize_text(prediction)
    if not pred:
        return False
    normalized = [normalize_text(answer) for answer in accepted]
    if pred in [answer for answer in normalized if answer]:
        return True
    if method == "numeric":
        numeric_prediction = _number(prediction)
        if numeric_prediction is None:
            return False
        return any(
            candidate is not None and candidate == numeric_prediction
            for candidate in (_number(answer) for answer in accepted)
        )
    return False


def _index(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if "id" not in row or row["id"] is None or not str(row["id"]).strip():
            raise ValueError(f"{label} contains a missing/empty id")
        key = str(row["id"]).strip()
        if key in indexed:
            raise ValueError(f"Duplicate {label} id: {key}")
        indexed[key] = row
    return indexed


def grade_predictions(
    questions: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    question_by_id = _index(questions, "question")
    prediction_by_id = _index(predictions, "prediction")
    if not question_by_id:
        raise ValueError("Cannot grade an empty question set")
    unexpected = set(prediction_by_id) - set(question_by_id)
    missing = set(question_by_id) - set(prediction_by_id)
    if unexpected:
        raise ValueError(f"Predictions contain unknown question IDs: {sorted(unexpected)}")
    if require_complete and missing:
        raise ValueError(f"Missing predictions for question IDs: {sorted(missing)}")
    rows = []
    for key, question in question_by_id.items():
        accepted = question.get("accepted_answers")
        if accepted is None:
            accepted = [question.get("expected_answer")]
        if isinstance(accepted, (str, bytes)) or not isinstance(accepted, (list, tuple)):
            raise ValueError(f"Question {key} must provide a list of accepted answers")
        if not any(normalize_text(answer) for answer in accepted):
            raise ValueError(f"Question {key} has no nonempty accepted answer")
        prediction = prediction_by_id.get(key, {})
        answer = prediction.get("model_answer", "")
        method = question.get("matching_method", "exact")
        rows.append({
            "id": question["id"],
            "category": question.get("category", "uncategorized"),
            "question": question["question"],
            "expected_answer": question.get("expected_answer", accepted[0]),
            "accepted_answers": list(accepted),
            "model_answer": answer,
            "is_correct": answer_is_correct(answer, accepted, method=method),
            "prediction_present": key in prediction_by_id,
            "matching_method": method,
            "evaluator_version": EVALUATOR_VERSION,
            "raw_output": prediction.get("raw_output", answer),
        })
    return rows


def summarize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    classification: str = "development",
    split_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(rows)
    _index(rows, "graded row")
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    if classification not in EVALUATION_CLASSIFICATIONS:
        raise ValueError(f"Unknown evaluation classification: {classification!r}")
    if classification == "held_out":
        if not split_verification or not (
            split_verification.get("verified") is True
            and split_verification.get("classification") == "held_out"
            and split_verification.get("evaluated_split") == "test"
            and split_verification.get("overlap_count") == 0
        ):
            raise ValueError("A held_out score requires a verified, disjoint frozen test manifest")
    if any(type(row.get("is_correct")) is not bool for row in rows):
        raise ValueError("is_correct must be a Boolean for every graded row")
    if any(row.get("evaluator_version") != EVALUATOR_VERSION for row in rows):
        raise ValueError("Rows must be regraded with the current evaluator before summarizing")
    complete = all(row.get("prediction_present", True) for row in rows)
    counts = Counter(row.get("category", "uncategorized") for row in rows)
    by_category = {}
    for category, total in counts.items():
        correct = sum(row["is_correct"] for row in rows if row.get("category", "uncategorized") == category)
        by_category[category] = {"total": total, "correct": correct, "accuracy": correct / total}
    correct = sum(row["is_correct"] for row in rows)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_classification": classification,
        "is_held_out_measurement": classification == "held_out" and complete,
        "complete": complete,
        "total_questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "by_category": by_category,
        "split_verification": dict(split_verification) if split_verification else None,
    }
