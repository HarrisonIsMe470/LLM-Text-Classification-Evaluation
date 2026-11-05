from types import SimpleNamespace

from data import data_loader


def test_remove_is_passed_through_without_forced_cleaning(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(data=["text"], target=[0], target_names=["label"])

    monkeypatch.setattr(data_loader, "fetch_20newsgroups", fake_fetch)

    result = data_loader.load_20newsgroups(
        subset="test",
        remove=("headers",),
        data_home=tmp_path,
    )

    assert result.data == ["text"]
    assert captured["remove"] == ("headers",)
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_empty_remove_stays_empty(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(data=[], target=[], target_names=[])

    monkeypatch.setattr(data_loader, "fetch_20newsgroups", fake_fetch)
    data_loader.load_20newsgroups(remove=(), data_home=tmp_path)

    assert captured["remove"] == ()
