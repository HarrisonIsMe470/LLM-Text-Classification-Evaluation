from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _build_aliases(labels: List[str]) -> List[str]:
    if len(labels) > 26:
        raise ValueError("Only supports up to 26 labels for A-Z aliases")
    return [chr(ord("A") + i) for i in range(len(labels))]


def _top_confusions(matrix: List[List[int]], labels: List[str], top_n: int = 10) -> List[Tuple[str, str, int]]:
    pairs: List[Tuple[str, str, int]] = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            if i == j:
                continue
            if count > 0:
                pairs.append((labels[i], labels[j], int(count)))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_n]


def _cell_style(value: int, column_max: int) -> str:
    if value <= 0 or column_max <= 0:
        return "background-color: #ffffff;"
    ratio = value / column_max
    # Larger value -> darker color.
    lightness = 97 - int(52 * ratio)
    return f"background-color: hsl(197, 82%, {lightness}%);"


def _matrix_html(matrix: List[List[int]], labels: List[str], aliases: List[str], title: str) -> str:
    n_cols = len(matrix[0]) if matrix else 0
    col_max = [0] * n_cols
    for row in matrix:
        for j, value in enumerate(row):
            if value > col_max[j]:
                col_max[j] = int(value)
    header_cells = "".join(f"<th>{html.escape(alias)}</th>" for alias in aliases)

    body_rows = []
    for i, row in enumerate(matrix):
        cells = []
        for j, value in enumerate(row):
            style = _cell_style(int(value), col_max[j])
            cells.append(f"<td style='{style}'>{int(value)}</td>")
        body_rows.append(f"<tr><th>{html.escape(aliases[i])}</th>{''.join(cells)}</tr>")

    legend = (
        "<div class='heat-legend'>"
        "<span>颜色深浅: 小</span>"
        "<span class='box light'></span>"
        "<span class='box mid'></span>"
        "<span class='box dark'></span>"
        "<span>大</span>"
        "</div>"
    )

    return (
        f"<h3>{html.escape(title)}</h3>"
        "<div class='table-wrap'><table class='matrix'>"
        f"<thead><tr><th>真实\\预测</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
        f"{legend}"
    )


def _alias_mapping_html(labels: List[str], aliases: List[str]) -> str:
    rows = []
    for alias, label in zip(aliases, labels):
        rows.append(f"<tr><td><b>{html.escape(alias)}</b></td><td>{html.escape(label)}</td></tr>")
    return (
        "<h3>类别缩写映射（A~T）</h3>"
        "<table class='alias-map'>"
        "<thead><tr><th>缩写</th><th>原始类别</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _find_pdf_browser() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    for path in candidates:
        if path.exists():
            return path
    for executable in ("msedge", "google-chrome", "chromium", "chromium-browser"):
        located = shutil.which(executable)
        if located:
            return Path(located)
    return None


def _write_html(report_html: Path, detailed: Dict[str, object], summary: Dict[str, object]) -> None:
    labels = list(summary["dataset"]["categories"])
    aliases = _build_aliases(labels)
    label_to_alias = {label: alias for label, alias in zip(labels, aliases)}

    tfidf = detailed["tfidf_logreg"]
    llm = detailed["llm_classifier"]
    dataset_info = summary.get("dataset", {})
    test_samples = dataset_info.get("num_test_samples", "N/A")

    llm_meta_path = PROJECT_ROOT / "modeling" / "configs" / "llm_run_metadata.json"
    llm_model_name = str(_load_json(llm_meta_path).get("model", "N/A")) if llm_meta_path.exists() else "N/A"
    tfidf_meta_path = PROJECT_ROOT / "modeling" / "configs" / "run_tfidf_logreg_metadata.json"
    tfidf_meta = _load_json(tfidf_meta_path) if tfidf_meta_path.exists() else {}
    train_samples = dataset_info.get("num_train_samples", tfidf_meta.get("train_size", "N/A"))

    tfidf_top_errors = _top_confusions(tfidf["confusion_matrix_20x20"], labels, top_n=12)
    llm_top_errors = _top_confusions(llm["confusion_matrix_20x20"], labels, top_n=12)

    def top_error_list(items: List[Tuple[str, str, int]]) -> str:
        lines = []
        for true_label, pred_label, count in items:
            true_alias = label_to_alias[true_label]
            pred_alias = label_to_alias[pred_label]
            lines.append(
                f"<li><code>{html.escape(true_alias)}</code> ({html.escape(true_label)}) 被预测为 "
                f"<code>{html.escape(pred_alias)}</code> ({html.escape(pred_label)}): <b>{count}</b> 次</li>"
            )
        return "<ol>" + "".join(lines) + "</ol>"

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>20 Newsgroups 分类评估报告</title>
  <style>
    @page {{ size: A4; margin: 14mm; }}
    body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: #111; line-height: 1.45; }}
    h1, h2, h3 {{ margin: 10px 0 6px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; border-left: 4px solid #0f766e; padding-left: 8px; }}
    h3 {{ font-size: 14px; }}
    p, li {{ font-size: 12px; }}
    code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 9px; table-layout: fixed; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 2px 3px; text-align: center; }}
    th {{ background: #e2e8f0; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 6px; }}
    .kpi {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
    .kpi th, .kpi td {{ font-size: 12px; padding: 6px; }}
    .alias-map th, .alias-map td {{ font-size: 11px; padding: 4px; text-align: left; }}
    .note {{ background: #f8fafc; border-left: 3px solid #0ea5e9; padding: 8px; font-size: 11px; }}
    .heat-legend {{ display: flex; align-items: center; gap: 6px; margin: 2px 0 10px; font-size: 10px; }}
    .heat-legend .box {{ width: 14px; height: 12px; border: 1px solid #94a3b8; display: inline-block; }}
    .heat-legend .light {{ background: hsl(197, 82%, 89%); }}
    .heat-legend .mid {{ background: hsl(197, 82%, 70%); }}
    .heat-legend .dark {{ background: hsl(197, 82%, 50%); }}
  </style>
</head>
<body>
  <h1>20 Newsgroups 文本分类评估报告</h1>
  <p>本报告由 <code>evaluation/evaluate_models.py</code> 与 <code>evaluation/generate_report.py</code> 自动生成。</p>

  <h2>1. 数据集说明</h2>
  <p><b>20 Newsgroups</b> 是一个英文新闻组文本分类数据集：每个样本是一条新闻组帖子文本；总样本数为 18846（训练集 <b>{train_samples}</b>、测试集 <b>{test_samples}</b>）；标签为 20 个主题新闻组类别。</p>
  <p>本项目在 <code>data/data_loader.py</code> 中通过 <code>fetch_20newsgroups</code> 加载该数据集，并按设置移除 headers/footers/quotes 以降低格式噪声。</p>
  <p>参数 <code>remove=('headers','footers','quotes')</code> 表示在加载阶段移除邮件头、邮件尾签名和历史引用段落，目标是降低与主题无关的格式噪声，避免模型利用发件信息或引用模板而非正文语义进行分类。</p>

  <h2>2. Baseline 与模型说明</h2>
  <ul>
    <li><b>tfidf_logreg</b>：<code>TfidfVectorizer + LogisticRegression</code> 的经典稀疏特征分类基线。模型可复用对象文件（例如向量化器、分类器、完整 pipeline）会在训练后保存到磁盘，路径记录在 <code>modeling/configs/run_tfidf_logreg_metadata.json</code> 中。</li>
    <li><b>llm_classifier</b>：基于 Hugging Face 因果语言模型的 zero-shot 分类器，当前记录模型为 <code>{html.escape(llm_model_name)}</code>。zero-shot 指不在本任务标签上进行参数微调，仅通过提示词让模型直接输出类别标签。</li>
  </ul>

  <h2>3. 配置说明</h2>
  <ul>
    <li>训练/推理入口：<code>main.py</code>（负责串联数据加载、baseline 训练、LLM 推理）。</li>
    <li>传统基线配置：<code>modeling/configs/run_tfidf_logreg_metadata.json</code>（记录样本规模、TF-IDF 参数、LogReg 参数、模型可复用对象文件路径与结果路径）。</li>
    <li>LLM 运行配置：<code>modeling/configs/llm_run_metadata.json</code>（记录模型名、设备、量化方式、输入截断阈值、batch size、zero-shot 汇总结果等）。</li>
    <li>评估输出目录：<code>evaluation/outputs</code>（统一存放模型对比评估所需的 JSON/CSV 结果文件）。</li>
  </ul>

  <h2>4. 全流程结构（Pipeline）</h2>
  <ol>
    <li>加载 20 Newsgroups 数据（训练/测试）</li>
    <li>训练 TF-IDF + Logistic Regression 基线并保存模型</li>
    <li>运行 LLM zero-shot 预测并保存 JSONL 结果</li>
    <li>统一评估脚本读取两种模型输出，计算宏平均与按类指标</li>
    <li>生成评估对象文件（JSON/CSV）并产出中文 PDF 报告</li>
  </ol>

  <h2>5. 输出说明</h2>
  <ul>
    <li><code>evaluation/outputs/metrics_summary.json</code>：模型级汇总指标（Macro-Precision / Macro-Recall / Macro-F1 / unknown 占比），用于快速横向对比。</li>
    <li><code>evaluation/outputs/detailed_metrics.json</code>：细粒度评估结果（逐类指标、混淆矩阵、unknown 计数等），用于诊断误差来源。</li>
    <li><code>evaluation/outputs/per_class_metrics.csv</code>：20 个类别逐类 Precision/Recall/F1，便于在电子表格中筛选排序。</li>
    <li><code>evaluation/outputs/confusion_matrix_*.csv</code>：混淆矩阵原始计数表；行表示真实标签，列表示预测标签，单元格为样本数。</li>
  </ul>

  <h2>6. 评估指标选择说明</h2>
  <p>本项目使用 <b>Macro-Precision / Macro-Recall / Macro-F1</b> 作为核心指标。含义为：先分别计算每个类别的 Precision/Recall/F1，再对 20 个类别做等权平均。这样不会因为某些类别样本更多而主导总分，能够更公平地反映模型在长尾类别上的识别能力。</p>

  <table class="kpi">
    <thead>
      <tr><th>模型</th><th>Macro-Precision</th><th>Macro-Recall</th><th>Macro-F1</th><th>未知预测占比</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>tfidf_logreg</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['macro_precision']))}</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['macro_recall']))}</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['macro_f1']))}</td>
        <td>{_fmt(float(summary['models']['tfidf_logreg']['unknown_prediction_rate']))}</td>
      </tr>
      <tr>
        <td>llm_classifier</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['macro_precision']))}</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['macro_recall']))}</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['macro_f1']))}</td>
        <td>{_fmt(float(summary['models']['llm_classifier']['unknown_prediction_rate']))}</td>
      </tr>
    </tbody>
  </table>
  <p>其中，LLM 预测中的 <code>unknown</code> 是指：模型输出未能解析为 20 个标准标签之一（例如输出自由文本解释、多个标签混合、或标签拼写/格式不匹配），因此在标签映射阶段被归入“不可映射预测”。</p>

  <h2>7. 混淆矩阵与误差分析</h2>
  <p>混淆矩阵用于展示“真实类别 vs 预测类别”的对应关系：对角线越大表示该类识别越准确，非对角线数值越大表示该真实类别更容易被误判到对应列类别。为提高可读性，图中使用 <b>A~T</b> 代替长类别名，完整映射见下方表格。</p>
  {_matrix_html(tfidf['confusion_matrix_20x20'], labels, aliases, 'TF-IDF + LogReg 混淆矩阵 (20x20)')}
  {_matrix_html(llm['confusion_matrix_20x20'], labels, aliases, 'LLM 混淆矩阵 (20x20，仅统计可映射标签)')}
  <div class="note">
    说明：在按 20 类混淆矩阵展示时， <code>unknown</code> 样本不计入 20x20 方阵，
    详细数量见 summary 文件中的 <code>unknown_prediction_rate</code>。
  </div>
  {_alias_mapping_html(labels, aliases)}

  <h3>TF-IDF + LogReg 主要混淆对</h3>
  {top_error_list(tfidf_top_errors)}

  <h3>LLM 主要混淆对</h3>
  {top_error_list(llm_top_errors)}

</body>
</html>
"""

    report_html.write_text(html_doc, encoding="utf-8")


def generate_report(
    eval_output_dir: Path,
    report_html: Path,
    report_pdf: Path,
    render_pdf: bool = True,
) -> Path:
    summary_path = eval_output_dir / "metrics_summary.json"
    detailed_path = eval_output_dir / "detailed_metrics.json"
    if not summary_path.exists() or not detailed_path.exists():
        raise FileNotFoundError(
            "Missing evaluation outputs. Run evaluation/evaluate_models.py first."
        )

    summary = _load_json(summary_path)
    detailed = _load_json(detailed_path)

    report_html.parent.mkdir(parents=True, exist_ok=True)
    _write_html(report_html, detailed=detailed, summary=summary)
    print(f"Saved HTML report: {report_html}")

    if not render_pdf:
        return report_html

    browser_path = _find_pdf_browser()
    if browser_path is None:
        print("Chrome/Edge/Chromium not found. Skipped PDF rendering.")
        return report_html

    report_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(browser_path),
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={report_pdf}",
        report_html.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved PDF report: {report_pdf}")
    return report_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chinese evaluation report HTML/PDF")
    parser.add_argument(
        "--eval-output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "outputs"),
        help="Directory containing evaluation outputs",
    )
    parser.add_argument(
        "--report-html",
        default=str(PROJECT_ROOT / "evaluation" / "report_zh.html"),
        help="Output HTML report path",
    )
    parser.add_argument(
        "--report-pdf",
        default=str(PROJECT_ROOT / "evaluation" / "report_zh.pdf"),
        help="Output PDF report path",
    )
    parser.add_argument(
        "--render-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render PDF with a locally installed Chrome, Edge, or Chromium browser",
    )
    args = parser.parse_args()
    generate_report(
        eval_output_dir=Path(args.eval_output_dir),
        report_html=Path(args.report_html),
        report_pdf=Path(args.report_pdf),
        render_pdf=args.render_pdf,
    )


if __name__ == "__main__":
    main()
