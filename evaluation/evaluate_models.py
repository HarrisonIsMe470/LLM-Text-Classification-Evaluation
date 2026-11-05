from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_20newsgroups


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_llm_predictions(path: Path, label_to_idx: Dict[str, int]) -> List[int]:
    predictions: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            predicted_label = row.get("predicted_label")
            if predicted_label is None:
                predictions.append(-1)
            else:
                predictions.append(label_to_idx.get(str(predicted_label), -1))
    return predictions


def _evaluate(
    name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, object]:
    class_indices = list(range(len(label_names)))

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_indices,
        average=None,
        zero_division=0,
    )

    unknown_col = len(label_names)
    matrix_20x21 = [[0 for _ in range(len(label_names) + 1)] for _ in range(len(label_names))]
    for true_idx, pred_idx in zip(y_true, y_pred):
        pred_col = pred_idx if 0 <= pred_idx < len(label_names) else unknown_col
        matrix_20x21[true_idx][pred_col] += 1

    y_pred_no_unknown = [p if 0 <= p < len(label_names) else -1 for p in y_pred]
    y_true_known: List[int] = []
    y_pred_known: List[int] = []
    for t, p in zip(y_true, y_pred_no_unknown):
        if p >= 0:
            y_true_known.append(t)
            y_pred_known.append(p)
    matrix_20x20 = confusion_matrix(
        y_true_known,
        y_pred_known,
        labels=class_indices,
    ).tolist()

    per_class_rows = []
    for idx, label in enumerate(label_names):
        per_class_rows.append(
            {
                "label": label,
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1": float(per_f1[idx]),
                "support": int(per_support[idx]),
            }
        )

    unknown_count = sum(1 for item in y_pred if item < 0 or item >= len(label_names))

    return {
        "model": name,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "sample_count": len(y_true),
        "unknown_prediction_count": unknown_count,
        "unknown_prediction_rate": float(unknown_count / len(y_true) if y_true else 0.0),
        "per_class": per_class_rows,
        "confusion_matrix_20x20": matrix_20x20,
        "confusion_matrix_20x21_with_unknown_col": matrix_20x21,
    }


def _write_per_class_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "label",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_matrix_csv(path: Path, label_names: Sequence[str], matrix: Sequence[Sequence[int]], include_unknown: bool) -> None:
    headers = ["true\\pred", *label_names]
    if include_unknown:
        headers.append("__UNKNOWN__")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx, row in enumerate(matrix):
            writer.writerow([label_names[idx], *row])


def _resolve_artifact_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evaluate_models(
    output_dir: Path,
    baseline_metadata_path: Path,
    llm_metadata_path: Path,
    llm_predictions_path: Optional[Path] = None,
    remove: Sequence[str] = (),
    max_test_samples: Optional[int] = None,
    test_dataset=None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_dataset is None:
        test_dataset = load_20newsgroups(subset="test", remove=remove)
    sample_count = len(test_dataset.data)
    if max_test_samples is not None:
        if max_test_samples <= 0:
            raise ValueError("max_test_samples must be a positive integer or None")
        sample_count = min(max_test_samples, sample_count)

    test_data = list(test_dataset.data[:sample_count])
    y_true = list(test_dataset.target[:sample_count])
    label_names = list(test_dataset.target_names)
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    baseline_metadata = _load_json(baseline_metadata_path)
    baseline_object = baseline_metadata.get("object")
    if not baseline_object:
        raise KeyError("Baseline metadata must contain 'object' field.")
    baseline_pipeline_path = _resolve_artifact_path(baseline_object["pipeline"])
    baseline_pipeline = joblib.load(baseline_pipeline_path)
    baseline_pred = list(baseline_pipeline.predict(test_data))

    if llm_predictions_path is None:
        llm_metadata = _load_json(llm_metadata_path)
        llm_predictions_path = _resolve_artifact_path(
            llm_metadata["zero_shot"]["predictions_path"]
        )
    llm_pred = _read_llm_predictions(llm_predictions_path, label_to_idx)

    if len(llm_pred) != len(y_true):
        raise ValueError(
            f"LLM prediction count ({len(llm_pred)}) does not match selected test "
            f"sample count ({len(y_true)})."
        )

    baseline_eval = _evaluate("tfidf_logreg", y_true, baseline_pred, label_names)
    llm_eval = _evaluate("llm_classifier", y_true, llm_pred, label_names)

    summary = {
        "dataset": {
            "name": "20 Newsgroups",
            "num_categories": len(label_names),
            "num_test_samples": len(y_true),
            "remove": list(remove),
            "categories": label_names,
        },
        "models": {
            "tfidf_logreg": {
                "macro_precision": baseline_eval["macro_precision"],
                "macro_recall": baseline_eval["macro_recall"],
                "macro_f1": baseline_eval["macro_f1"],
                "unknown_prediction_count": baseline_eval["unknown_prediction_count"],
                "unknown_prediction_rate": baseline_eval["unknown_prediction_rate"],
            },
            "llm_classifier": {
                "macro_precision": llm_eval["macro_precision"],
                "macro_recall": llm_eval["macro_recall"],
                "macro_f1": llm_eval["macro_f1"],
                "unknown_prediction_count": llm_eval["unknown_prediction_count"],
                "unknown_prediction_rate": llm_eval["unknown_prediction_rate"],
            },
        },
        "object": {
            "baseline_pipeline": _portable_path(baseline_pipeline_path),
            "llm_predictions": _portable_path(llm_predictions_path),
        },
    }

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    detailed_path = output_dir / "detailed_metrics.json"
    with detailed_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": summary["dataset"],
                "tfidf_logreg": baseline_eval,
                "llm_classifier": llm_eval,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    per_class_rows = []
    for model_name, eval_obj in (
        ("tfidf_logreg", baseline_eval),
        ("llm_classifier", llm_eval),
    ):
        for row in eval_obj["per_class"]:
            per_class_rows.append({"model": model_name, **row})
    _write_per_class_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    _write_matrix_csv(
        output_dir / "confusion_matrix_tfidf_logreg_20x20.csv",
        label_names,
        baseline_eval["confusion_matrix_20x20"],
        include_unknown=False,
    )
    _write_matrix_csv(
        output_dir / "confusion_matrix_llm_classifier_20x20.csv",
        label_names,
        llm_eval["confusion_matrix_20x20"],
        include_unknown=False,
    )
    _write_matrix_csv(
        output_dir / "confusion_matrix_llm_classifier_20x21_with_unknown.csv",
        label_names,
        llm_eval["confusion_matrix_20x21_with_unknown_col"],
        include_unknown=True,
    )

    print(f"Saved: {summary_path}")
    print(f"Saved: {detailed_path}")
    print(f"Saved: {output_dir / 'per_class_metrics.csv'}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TF-IDF+LogReg and LLM classifier with macro Precision/Recall/F1 "
            "and per-class metrics over all 20 categories."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "outputs"),
        help="Directory to save evaluation objects",
    )
    parser.add_argument(
        "--baseline-metadata",
        default=str(PROJECT_ROOT / "modeling" / "configs" / "run_tfidf_logreg_metadata.json"),
        help="Path to baseline metadata JSON",
    )
    parser.add_argument(
        "--llm-metadata",
        default=str(PROJECT_ROOT / "modeling" / "configs" / "llm_run_metadata.json"),
        help="Path to LLM metadata JSON",
    )
    parser.add_argument(
        "--llm-predictions",
        default="",
        help="Optional override for LLM predictions JSONL path",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        choices=["headers", "footers", "quotes"],
        help="Elements removed when loading the test split",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Evaluate only the first N deterministic test samples",
    )
    args = parser.parse_args()
    evaluate_models(
        output_dir=Path(args.output_dir),
        baseline_metadata_path=Path(args.baseline_metadata),
        llm_metadata_path=Path(args.llm_metadata),
        llm_predictions_path=Path(args.llm_predictions) if args.llm_predictions else None,
        remove=args.remove,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()

