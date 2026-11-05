from __future__ import annotations

import argparse
from pathlib import Path

from data.data_loader import load_20newsgroups
from evaluation.evaluate_models import evaluate_models
from evaluation.generate_report import generate_report
from modeling.baseline_tfidf_logreg import train_and_evaluate
from modeling.llm_classifier import run_llm_baseline


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entry point: load dataset, run baseline + LLM prediction, and save results",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=["headers", "footers", "quotes"],
        choices=["headers", "footers", "quotes"],
        help="Elements to remove from the text",
    )
    parser.add_argument(
        "--llm-max-test-samples",
        type=int,
        default=None,
        help="Limit LLM/evaluation samples; defaults to the full test split",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--llm-max-chars",
        type=int,
        default=1600,
        help="Max characters per document before prompt construction",
    )
    parser.add_argument(
        "--llm-max-input-tokens",
        type=int,
        default=512,
        help="Tokenizer truncation limit for prompt inputs",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=32,
        help="Generation batch size; automatically reduces after CUDA OOM",
    )
    parser.add_argument(
        "--llm-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. 'auto' falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--llm-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit loading on CUDA; automatically disabled on CPU",
    )
    parser.add_argument(
        "--llm-quiet-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress Hugging Face logs/progress (default: show progress to avoid silent startup)",
    )
    parser.add_argument(
        "--render-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render PDF when Chrome, Edge, or Chromium is installed",
    )
    args = parser.parse_args()

    train_dataset = load_20newsgroups(subset="train", remove=args.remove)
    test_dataset = load_20newsgroups(subset="test", remove=args.remove)

    print(f"Train documents: {len(train_dataset.data)}", flush=True)
    print(f"Test documents: {len(test_dataset.data)}", flush=True)
    print(f"Categories: {len(train_dataset.target_names)}", flush=True)

    print("[tfidf-logreg] Starting baseline prediction...", flush=True)
    baseline_metadata = train_and_evaluate(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_dir=PROJECT_ROOT / "modeling" / "configs",
        output_dir=OUTPUTS_DIR,
        metadata_dir=PROJECT_ROOT / "modeling" / "configs",
    )
    print(
        "[tfidf-logreg] Completed. "
        f"Metadata: {PROJECT_ROOT / 'modeling' / 'configs' / 'run_tfidf_logreg_metadata.json'}",
        flush=True,
    )
    print(
        f"[tfidf-logreg] Accuracy: {baseline_metadata['metrics']['accuracy']:.4f}",
        flush=True,
    )

    print("[llm] Starting LLM prediction...", flush=True)

    llm_metadata_path = run_llm_baseline(
        max_test_samples=args.llm_max_test_samples,
        model=args.llm_model,
        max_chars=args.llm_max_chars,
        max_input_tokens=args.llm_max_input_tokens,
        batch_size=args.llm_batch_size,
        device=args.llm_device,
        load_in_4bit=args.llm_load_in_4bit,
        quiet_hf=args.llm_quiet_hf,
        output_dir=OUTPUTS_DIR,
        metadata_dir=PROJECT_ROOT / "modeling" / "configs",
        remove=args.remove,
        test_dataset=test_dataset,
    )
    print(f"[llm] Completed. Metadata: {llm_metadata_path}", flush=True)

    print("[evaluation] Computing unified metrics...", flush=True)
    evaluate_models(
        output_dir=PROJECT_ROOT / "evaluation" / "outputs",
        baseline_metadata_path=PROJECT_ROOT / "modeling" / "configs" / "run_tfidf_logreg_metadata.json",
        llm_metadata_path=llm_metadata_path,
        remove=args.remove,
        max_test_samples=args.llm_max_test_samples,
        test_dataset=test_dataset,
    )

    print("[report] Generating evaluation report...", flush=True)
    generate_report(
        eval_output_dir=PROJECT_ROOT / "evaluation" / "outputs",
        report_html=PROJECT_ROOT / "evaluation" / "report_zh.html",
        report_pdf=PROJECT_ROOT / "evaluation" / "report_zh.pdf",
        render_pdf=args.render_pdf,
    )
    print("[pipeline] Complete.", flush=True)


if __name__ == "__main__":
    main()
