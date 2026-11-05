from modeling import llm_classifier


def test_device_auto_falls_back_to_cpu():
    assert llm_classifier._resolve_device("auto", cuda_available=False) == "cpu"
    assert llm_classifier._resolve_device("auto", cuda_available=True) == "cuda"


def test_explicit_cuda_requires_cuda():
    try:
        llm_classifier._resolve_device("cuda", cuda_available=False)
    except RuntimeError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("Expected unavailable CUDA to fail")


def test_evaluate_respects_batch_and_token_limits(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append((kwargs["batch_size"], kwargs["max_input_tokens"]))
        return ["a"] * len(kwargs["prompts"])

    monkeypatch.setattr(llm_classifier, "_generate_labels_batch", fake_generate)

    metrics, predictions, _, _ = llm_classifier._evaluate(
        tokenizer=object(),
        model=object(),
        eval_name="test",
        label_names=["a", "b"],
        texts=["x", "y", "z"],
        targets=[0, 0, 0],
        prompt_builder=lambda text: text,
        temperature=0.0,
        max_output_tokens=4,
        max_chars=10,
        batch_size=2,
        token_monitor_every=0,
        max_input_tokens=7,
        prompts=["x", "y", "z"],
        prompt_token_lengths=[1, 1, 1],
    )

    assert calls == [(2, 7), (1, 7)]
    assert predictions == ["a", "a", "a"]
    assert metrics["fixed_normal_batch_size"] == 2


def test_artifact_metadata_path_is_project_relative():
    path = llm_classifier.PROJECT_ROOT / "outputs" / "predictions.jsonl"
    assert llm_classifier._artifact_path(path) == "outputs/predictions.jsonl"
