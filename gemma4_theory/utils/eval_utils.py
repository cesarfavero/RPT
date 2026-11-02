import json
from datetime import datetime
from pathlib import Path
import sys

# One implementation for notebooks, development regrading, and validation CLI.
# Run from a repository checkout so the matching policy cannot silently drift.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.evaluation import (  # noqa: E402,F401
    EVALUATOR_VERSION,
    answer_is_correct,
    first_sentence,
    grade_predictions,
    normalize_text,
    summarize_rows,
)


def load_questions(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamped_result_path(results_dir, stem):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(results_dir) / f'{stem}_{ts}.json'

