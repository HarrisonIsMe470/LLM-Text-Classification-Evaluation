import json
from types import SimpleNamespace

from evaluation import evaluate_models as module


class FakePipeline:
    def predict(self, data):
        return [0 for _ in data]


def test_evaluation_accepts_a_matching_smoke_subset(monkeypatch, tmp_path):
    baseline_metadata = tmp_path / "baseline.json"
    llm_metadata = tmp_path / "llm.json"
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "evaluation"

    baseline_metadata.write_text(
        json.dumps({"object": {"pipeline": "modeling/configs/model.joblib"}}),
        encoding="utf-8",
    )
    llm_metadata.write_text(
        json.dumps({"zero_shot": {"predictions_path": str(predictions)}}),
        encoding="utf-8",
    )
    predictions.write_text(
        "\n".join(
            json.dumps({"predicted_label": label})
            for label in ("a", "a")
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        data=["one", "two", "three"],
        target=[0, 1, 1],
        target_names=["a", "b"],
    )
    monkeypatch.setattr(module.joblib, "load", lambda _: FakePipeline())

    summary_path = module.evaluate_models(
        output_dir=output_dir,
        baseline_metadata_path=baseline_metadata,
        llm_metadata_path=llm_metadata,
        max_test_samples=2,
        test_dataset=dataset,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset"]["num_test_samples"] == 2
    assert summary["object"]["llm_predictions"] == str(predictions)
