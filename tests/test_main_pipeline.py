import sys
from types import SimpleNamespace

import main as pipeline


def test_main_orchestrates_inference_evaluation_and_report(monkeypatch, tmp_path):
    events = []
    dataset = SimpleNamespace(
        data=["one", "two", "three"],
        target=[0, 1, 0],
        target_names=["a", "b"],
    )

    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(
        pipeline,
        "load_20newsgroups",
        lambda subset, remove: dataset,
    )
    monkeypatch.setattr(
        pipeline,
        "train_and_evaluate",
        lambda **kwargs: {"metrics": {"accuracy": 1.0}},
    )

    metadata_path = tmp_path / "modeling" / "configs" / "llm_run_metadata.json"

    def fake_llm(**kwargs):
        events.append(("llm", kwargs["max_test_samples"]))
        return metadata_path

    def fake_evaluate(**kwargs):
        events.append(("evaluate", kwargs["max_test_samples"]))

    def fake_report(**kwargs):
        events.append(("report", kwargs["render_pdf"]))

    monkeypatch.setattr(pipeline, "run_llm_baseline", fake_llm)
    monkeypatch.setattr(pipeline, "evaluate_models", fake_evaluate)
    monkeypatch.setattr(pipeline, "generate_report", fake_report)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--llm-max-test-samples", "2", "--no-render-pdf"],
    )

    pipeline.main()

    assert events == [("llm", 2), ("evaluate", 2), ("report", False)]
