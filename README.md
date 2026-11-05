# LLM Text Classification Evaluation

A reproducible evaluation pipeline for comparing two text-classification approaches on the `20 Newsgroups` dataset:

- **Supervised classical baseline:** TF-IDF + Logistic Regression trained on the full training split.
- **Small zero-shot LLM baseline:** `Qwen/Qwen2.5-1.5B-Instruct`, evaluated without task-specific training examples.

The project examines quality, invalid-label behavior, latency, and class-level failure patterns under two different learning and cost settings. Because one model receives supervised training while the other is evaluated zero-shot, the results must **not** be interpreted as a fair capability ranking between classical machine learning and LLMs.

The main pipeline loads the data, trains the supervised baseline, runs LLM inference, computes unified metrics, and generates an HTML/PDF report.

## Reference results

The following results came from one full test-split run with `headers`, `footers`, and `quotes` removed:

| Model | Macro-Precision | Macro-Recall | Macro-F1 | Invalid-label rate |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.6653 | 0.6469 | 0.6450 | 0% |
| Qwen2.5-1.5B-Instruct zero-shot | 0.3580 | 0.1865 | 0.2001 | 27.75% |

These values are historical reference results rather than committed runtime artifacts. Each run regenerates its own metrics, predictions, configuration metadata, and timing information.

## Requirements

- Python 3.11 or later.
- CPU is supported for a small smoke run.
- An NVIDIA CUDA GPU is recommended for the full LLM evaluation.
- Direct dependencies are pinned in `requirements.txt`.

GPU users should first install the PyTorch build that matches their driver using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the remaining pinned dependencies.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dataset is downloaded by scikit-learn into the Git-ignored `data/cache/` directory. It can also be cached in advance:

```bash
python scripts/download_data.py
```

## CPU smoke run

This command processes 20 LLM test samples and verifies the complete pipeline without CUDA. The first run downloads the 0.5B model checkpoint.

```bash
python main.py \
  --remove headers footers quotes \
  --llm-device cpu \
  --llm-model Qwen/Qwen2.5-0.5B-Instruct \
  --llm-max-test-samples 20 \
  --llm-batch-size 1 \
  --llm-max-chars 800 \
  --llm-max-input-tokens 256 \
  --no-llm-load-in-4bit \
  --no-render-pdf
```

The smoke run still trains the full TF-IDF baseline, but LLM inference and unified evaluation use the same deterministic 20-sample test subset. This prevents prediction-count mismatches between limited and full runs.

## Full GPU run

First confirm that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`, then run:

```bash
python main.py \
  --remove headers footers quotes \
  --llm-device cuda \
  --llm-max-test-samples 7532 \
  --llm-batch-size 32 \
  --llm-max-chars 1600 \
  --llm-max-input-tokens 512 \
  --llm-load-in-4bit
```

The default device is `auto`: CUDA is used when available, otherwise inference falls back to CPU. The supplied character, token, and batch limits are respected. After a CUDA out-of-memory error, only the current retry batch is reduced.

## Generated outputs

Runtime artifacts are excluded by `.gitignore`:

- `modeling/configs/`: trained models and metadata containing portable project-relative paths.
- `outputs/`: per-sample LLM predictions, raw model output, and timing summaries.
- `evaluation/outputs/`: macro metrics, per-class metrics, and confusion matrices.
- `evaluation/report_zh.html`: always generated.
- `evaluation/report_zh.pdf`: generated when Chrome, Edge, or Chromium is installed.

Individual stages can also be run separately:

```bash
python modeling/baseline_tfidf_logreg.py --remove headers footers quotes
python modeling/llm_classifier.py --remove headers footers quotes --device auto
python evaluation/evaluate_models.py --remove headers footers quotes
python evaluation/generate_report.py
```

## Testing and CI

The lightweight test suite does not download the dataset or model:

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
```

GitHub Actions runs the same tests on Python 3.11. The suite covers preprocessing arguments, CPU fallback, batch and token limits, portable artifact paths, subset-aligned evaluation, and main-pipeline orchestration.

## Project structure

```text
.
├── main.py
├── data/data_loader.py
├── modeling/
│   ├── baseline_tfidf_logreg.py
│   └── llm_classifier.py
├── evaluation/
│   ├── evaluate_models.py
│   └── generate_report.py
├── scripts/download_data.py
├── tests/
└── .github/workflows/ci.yml
```

## License

MIT License. Author: Haochen.
