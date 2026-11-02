"""Prevent notebook drift from reintroducing permissive copied judges."""
import ast
import json
from pathlib import Path
import unittest

from validation.evaluation import answer_is_correct

ROOT = Path(__file__).resolve().parents[1]


class NotebookEvaluatorTests(unittest.TestCase):
    def test_no_embedded_legacy_evaluator(self):
        found = 0
        for path in (ROOT / "gemma4_theory/notebooks").glob("*.ipynb"):
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
                source = "".join(cell.get("source", []))
                if cell["cell_type"] != "code":
                    continue
                self.assertNotIn("utils_payload =", source, path.name)
                self.assertNotIn("e in a or a in e", source, path.name)
                self.assertNotIn("expected in actual or actual in expected", source, path.name)
                if "def judge_answer(" in source:
                    tree = ast.parse(source)
                    judge = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "judge_answer")
                    namespace = {"answer_is_correct": answer_is_correct}
                    exec(compile(ast.Module(body=[judge], type_ignores=[]), path.name, "exec"), namespace)
                    for actual in ("", "Não é Paris", "Paris, mas é Berlim"):
                        self.assertFalse(namespace["judge_answer"]({"expected_answer": "Paris"}, actual), path.name)
                    self.assertTrue(namespace["judge_answer"]({"expected_answer": "Paris"}, "Paris."))
                    found += 1
        self.assertGreaterEqual(found, 6)


if __name__ == "__main__":
    unittest.main()
