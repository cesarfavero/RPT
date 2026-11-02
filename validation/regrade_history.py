"""Regrade saved historical answers without regenerating or editing source results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from validation.evaluation import grade_predictions, summarize_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "gemma4_theory/results/reference_eval/16_reference_hf_gemma_2_2b_it_original_20260422_181209.json",
    "gemma4_theory/results/reference_eval/16_reference_student_pruned_20260422_181457.json",
    "gemma4_theory/results/packed_recovery/18_test_packed_recovered_l78_20260422_205445.json",
    "gemma4_theory/results/packed_recovery/24_integrate_triton_ternary_runtime_20260424_192203.json",
]


def regrade(source, questions):
    data = json.loads(source.read_text(encoding="utf-8"))
    original = data.get("rows", data.get("eval_rows"))
    if not isinstance(original, list):
        raise ValueError("Historical result has no rows/eval_rows")
    by_question = {q["question"]: q for q in questions}
    if len(by_question) != len(questions):
        raise ValueError("Historical questions are not unique")
    predictions, old_by_id = [], {}
    for row in original:
        question = by_question.get(row["question"])
        if question is None:
            raise ValueError("Historical question differs from declared suite")
        if "id" in row and str(row["id"]) != str(question["id"]):
            raise ValueError("Historical ID does not match question")
        answer = row.get("model_answer", row.get("answer", ""))
        predictions.append({"id": question["id"], "model_answer": answer})
        old = row.get("is_correct", row.get("ok"))
        if type(old) is not bool:
            raise ValueError("Historical result is missing a Boolean judgment")
        old_by_id[str(question["id"])] = old
    rows = grade_predictions(questions, predictions)
    changed = [{"id": row["id"], "question": row["question"], "model_answer": row["model_answer"],
                "accepted_answers": row["accepted_answers"], "old_is_correct": old_by_id[str(row["id"])],
                "new_is_correct": row["is_correct"]}
               for row in rows if old_by_id[str(row["id"])] != row["is_correct"]]
    return {"source": source.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "historical_correct": sum(old_by_id.values()), "summary": summarize_rows(rows),
            "changed_judgments": changed, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new output; historical and derived results are immutable")
    questions_path = ROOT / "gemma4_theory/data/gemma4_100_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    report = {"schema_version": "rpt-historical-regrade-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_classification": "development", "model_executed": False,
              "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
              "scope": "Saved answers rescored under a stricter full-answer rule; not new generations or independent generalization evidence.",
              "runs": [regrade(ROOT / path, questions) for path in SOURCES]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "scores": [
        {"source": item["source"], "old": item["historical_correct"], "strict": item["summary"]["correct"]}
        for item in report["runs"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
