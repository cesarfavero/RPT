import json
from pathlib import Path
import tempfile
import unittest

from validation.splits import (
    HISTORICAL_QUESTIONS, SplitValidationError, build_split_manifest, find_overlaps,
    freeze_split_manifest, load_records, prompt_sha256, record_fingerprints,
    verify_split_manifest, write_frozen_manifest,
)


class SplitIdentityTests(unittest.TestCase):
    def test_prompt_normalization_catches_case_accents_whitespace_punctuation(self):
        self.assertEqual(prompt_sha256("  Qual É a capital ? \n"), prompt_sha256("qual e a capital?"))

    def test_same_prompt_with_new_id_is_still_overlap(self):
        with self.assertRaises(SplitValidationError):
            build_split_manifest({"train": [{"id": 1, "question": "A new question?"}],
                                  "test": [{"id": 999, "question": " A NEW QUESTION ? "}]},
                                 dataset_id="experiment", include_historical_exposure=False)

    def test_same_id_with_changed_prompt_is_detected(self):
        with self.assertRaises(SplitValidationError):
            build_split_manifest({"train": [{"id": 1, "question": "First question"}],
                                  "test": [{"id": 1, "question": "Different question"}]},
                                 dataset_id="experiment", include_historical_exposure=False)

    def test_dataset_namespaces_disambiguate_local_ids(self):
        manifest = build_split_manifest({"train": [{"id": 1, "dataset_id": "one", "question": "First question"}],
                                         "test": [{"id": 1, "dataset_id": "two", "question": "Second question"}]},
                                        dataset_id="experiment", include_historical_exposure=False)
        self.assertEqual(manifest["overlaps"], [])

    def test_historical_training_questions_cannot_be_relabelled_test(self):
        question = load_records(HISTORICAL_QUESTIONS)[0]
        question["id"] = "new-looking-id"
        with self.assertRaises(SplitValidationError):
            build_split_manifest({"test": [question]}, dataset_id="new-file", training_data_complete=True)

    def test_historical_questions_are_allowed_as_development(self):
        manifest = build_split_manifest({"development": load_records(HISTORICAL_QUESTIONS)}, dataset_id="historical")
        self.assertEqual(manifest["splits"]["development"]["count"], 100)

    def test_additional_training_source_cannot_overlap_test(self):
        with self.assertRaises(SplitValidationError):
            build_split_manifest({"test": [{"id": "t", "prompt": "Previously exposed question"}]}, dataset_id="new",
                                 known_training_records=[{"id": "a", "text": "previously EXPOSED question?"}],
                                 include_historical_exposure=False)

    def test_multiple_choice_identity_does_not_hide_raw_question_overlap(self):
        record = {"id": 1, "question": "Known question?", "rendered_prompt": "Question: Known question?\nAnswer:",
                  "fingerprint_prompt": '{"question":"Known question?","choices":["a","b"]}'}
        with self.assertRaises(SplitValidationError):
            build_split_manifest({"test": [record]}, dataset_id="external", prompt_field="fingerprint_prompt",
                                 known_training_records=[{"question": "Known question?"}], include_historical_exposure=False)

    def test_duplicate_ids_and_duplicate_prompts_fail(self):
        for records in ([{"id": 1, "question": "One"}, {"id": 1, "question": "Two"}],
                        [{"id": 1, "question": "One"}, {"id": 2, "question": "ONE"}]):
            with self.subTest(records=records), self.assertRaises(SplitValidationError):
                record_fingerprints(records, dataset_id="duplicates")

    def test_development_overlap_requires_explicit_permission(self):
        records = {"train": [{"id": 1, "question": "Reusable example"}],
                   "development": [{"id": 2, "question": "Reusable example"}]}
        with self.assertRaises(SplitValidationError):
            build_split_manifest(records, dataset_id="dev", include_historical_exposure=False)
        manifest = build_split_manifest(records, dataset_id="dev", allow_train_development_overlap=True,
                                        include_historical_exposure=False)
        self.assertEqual(len(manifest["overlaps"]), 1)


class FrozenManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.json"
        self.test = self.root / "test.json"
        self.manifest = self.root / "manifest.json"
        self.train.write_text(json.dumps([{"id": 1, "question": "An unseen training prompt"}]))
        self.test.write_text(json.dumps([{"id": 2, "question": "An independent evaluation prompt"}]))

    def freeze(self, complete=False):
        return freeze_split_manifest({"train": self.train, "test": self.test}, self.manifest,
                                     dataset_id="test-experiment", training_data_complete=complete)

    def test_verified_complete_training_supports_scoped_heldout(self):
        manifest = self.freeze(complete=True)
        report = verify_split_manifest(self.manifest, expected_sha256=manifest["manifest_sha256"],
                                       records=load_records(self.test))
        self.assertEqual(report["classification"], "held_out")
        self.assertEqual(report["overlap_count"], 0)
        self.assertIn("pretraining", report["held_out_scope"])
        self.assertFalse(report["semantic_overlap_checked"])

    def test_unknown_training_exposure_stays_external_validation(self):
        self.freeze()
        self.assertEqual(verify_split_manifest(self.manifest)["classification"], "external_validation")

    def test_freeze_refuses_to_overwrite(self):
        self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_changed_dataset_is_rejected(self):
        self.freeze()
        self.test.write_text(json.dumps([{"id": 2, "question": "Altered after declaration"}]))
        with self.assertRaises(SplitValidationError):
            verify_split_manifest(self.manifest)

    def test_changed_answer_is_also_detected(self):
        self.freeze()
        records = load_records(self.test)
        records[0]["expected_answer"] = "changed answer"
        with self.assertRaises(SplitValidationError):
            verify_split_manifest(self.manifest, records=records)

    def test_changed_manifest_is_rejected(self):
        self.freeze()
        manifest = json.loads(self.manifest.read_text())
        manifest["training_data_complete"] = True
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaises(SplitValidationError):
            verify_split_manifest(self.manifest)

    def test_wrong_expected_hash_is_rejected(self):
        self.freeze()
        with self.assertRaises(SplitValidationError):
            verify_split_manifest(self.manifest, expected_sha256="0" * 64)

    def test_reordered_records_cannot_silently_change_limit_selection(self):
        self.test.write_text(json.dumps([{"id": 2, "question": "Second"}, {"id": 3, "question": "Third"}]))
        self.freeze()
        with self.assertRaises(SplitValidationError):
            verify_split_manifest(self.manifest, records=list(reversed(load_records(self.test))))

    def test_missing_source_fails(self):
        self.freeze()
        self.train.unlink()
        with self.assertRaises(SplitValidationError):
            verify_split_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
