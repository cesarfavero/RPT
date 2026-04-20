import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_questions(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(text):
    text = '' if text is None else str(text)
    text = text.strip().lower()
    text = text.replace('**', '').replace('`', '')
    text = text.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('á', 'a').replace('à', 'a').replace('ã', 'a').replace('â', 'a')
    text = text.replace('é', 'e').replace('ê', 'e')
    text = text.replace('í', 'i')
    text = text.replace('ó', 'o').replace('ô', 'o').replace('õ', 'o')
    text = text.replace('ú', 'u')
    text = text.replace('ç', 'c')
    return text


def first_sentence(text):
    text = (text or '').strip()
    parts = re.split(r'(?<=[\.!?])\s+', text)
    return parts[0] if parts and parts[0] else text


def answer_is_correct(prediction, accepted_answers):
    pred = normalize_text(first_sentence(prediction))
    accepted = [normalize_text(a) for a in accepted_answers]
    if pred in accepted:
        return True
    return any(ans in pred for ans in accepted if ans)


def grade_predictions(questions, predictions):
    by_id = {int(p['id']): p for p in predictions}
    rows = []
    for q in questions:
        pred = by_id.get(int(q['id']), {})
        answer = pred.get('model_answer', '')
        correct = answer_is_correct(answer, q['accepted_answers'])
        rows.append({
            'id': q['id'],
            'category': q['category'],
            'question': q['question'],
            'expected_answer': q['expected_answer'],
            'model_answer': answer,
            'is_correct': correct,
            'raw_output': pred.get('raw_output', answer),
        })
    return rows


def summarize_rows(rows):
    total = len(rows)
    correct = sum(1 for r in rows if r['is_correct'])
    by_cat = {}
    counts = Counter(r['category'] for r in rows)
    for category, n in counts.items():
        cat_rows = [r for r in rows if r['category'] == category]
        cat_correct = sum(1 for r in cat_rows if r['is_correct'])
        by_cat[category] = {
            'total': n,
            'correct': cat_correct,
            'accuracy': cat_correct / n if n else 0.0,
        }
    return {
        'total_questions': total,
        'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'by_category': by_cat,
    }


def save_json(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def timestamped_result_path(results_dir, stem):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(results_dir) / f'{stem}_{ts}.json'
