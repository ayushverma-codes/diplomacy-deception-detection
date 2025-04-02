# Diplomacy Deception Detection

A research-inspired NLP project that investigates **actual deception** and **suspected deception** in conversations from the strategy game *Diplomacy*, using traditional machine learning, recurrent neural networks, conversation context, power dynamics, and BERT.

Inspired by [*It Takes Two to Lie: One to Lie, and One to Listen* (ACL 2020)](https://aclanthology.org/2020.acl-main.353/).

## Overview

In *Diplomacy*, players negotiate alliances, coordinate moves, and sometimes mislead one another. This makes the game a useful setting for studying deception in natural-language conversations.

The project treats deception as **two separate binary classification tasks**:

- **Actual lie detection:** Was the sender's message truthful or deceptive, according to the sender's annotation?
- **Suspected lie detection:** Did the receiver perceive the message as truthful or deceptive?

These labels represent different perspectives and are modeled separately.

## Project Objectives

- Validate, preprocess, and explore the Diplomacy conversation dataset.
- Build traditional machine-learning baselines and neural classifiers.
- Study whether conversation context and relative player power improve predictions.
- Evaluate BERT-based message representations.
- Compare models and analyze their errors, ablations, and limitations.

## Dataset

The dataset contains **17,289 messages from 12 Diplomacy games**, using the original **game-level train/validation/test split** (9/1/2 games). Messages include conversation text, game and participant information, chronological context, and sender/receiver annotations. Relative player scores support power-dynamics features.

The original files remain in `data/train.jsonl`, `data/validation.jsonl`, and `data/test.jsonl`. For suspected-lie training and evaluation, receiver labels marked `NOANNOTATION` are excluded from supervision rather than treated as negative examples; unannotated messages may still provide conversational context.

## Implementation and Experiments

| Phase | Work completed |
|---|---|
| **1 — Data preparation and EDA** | JSONL loading, schema validation, preprocessing, conversation construction, and dataset/class-distribution analysis. |
| **2 — Traditional ML** | Random and majority baselines; Harbingers, Logistic Regression, Bag-of-Words, and power-enhanced feature combinations. |
| **3 — Standard LSTM** | Message-level bidirectional LSTM using pretrained GloVe embeddings, trained separately for both prediction tasks. |
| **4 — Context-aware neural models** | Message-level BiLSTM, BiLSTM with concatenated preceding messages, and hierarchical/context LSTM. |
| **5 — Hierarchical LSTM extensions** | Hierarchical LSTM, +Power, +BERT, and +BERT+Power; eight model/task experiments in total. |
| **6 — Analysis and validation** | BERT failure diagnostics, ablations, error analysis, interpretability-oriented analysis, historical benchmarking, artifact checks, and research validation. |

The Phase 4 BiLSTM overlaps architecturally with the earlier message-level LSTM; some context-model details are adaptations rather than exact reproductions of the paper.

## Model Performance

**Phase 5 results on the original held-out test split** (two games):

| Model | Actual Lie Macro F1 | Actual Lie F1 | Suspected Lie Macro F1 | Suspected Lie F1 |
|---|---:|---:|---:|---:|
| Hierarchical LSTM | 52.63% | 10.94% | 50.93% | 6.98% |
| **Hierarchical LSTM + Power** | **57.95%** | **24.06%** | **53.36%** | **13.26%** |
| Hierarchical LSTM + BERT | 47.71% | 0.00% | 48.28% | 0.00% |
| Hierarchical LSTM + BERT + Power | 47.71% | 0.00% | 48.28% | 0.00% |

**Hierarchical LSTM + Power** achieved the highest Phase 5 Macro F1 and positive-class F1 on both tasks. At the default decision threshold, both BERT-based variants predicted only the majority class. These results describe a small two-game test split and **do not establish statistical significance**.

For detailed comparisons, see [`docs/phase_5_results_comparison.md`](docs/phase_5_results_comparison.md) and [`docs/final_research_report.md`](docs/final_research_report.md).

## Technology Stack

- **Language:** Python
- **Machine learning:** PyTorch, scikit-learn
- **NLP:** Hugging Face Transformers, pretrained GloVe embeddings
- **Data and analysis:** pandas, NumPy, SciPy, Matplotlib, Seaborn
- **Experimentation:** Jupyter Notebook, Google Colab, YAML configurations
- **Testing:** pytest

## Project Structure

```text
diplomacy-deception-detection/
├── configs/          # Experiment settings
├── data/             # Original train/validation/test JSONL files
├── docs/             # Phase reports and research findings
├── notebooks/        # EDA and Colab experiment notebooks
├── src/diplomacy/    # Data, features, models, training and analysis code
├── tests/            # Automated tests
├── artifacts/        # Generated checkpoints, predictions, metrics and plots
├── requirements.txt
└── README.md
```

Generated `artifacts/` content, including large model checkpoints, is normally excluded from Git. Restore the required artifacts separately to reproduce saved-model inference and artifact-based analyses.

## Running the Project

Clone your copy of the repository and install the dependencies:

```bash
git clone <your-repository-url>
cd diplomacy-deception-detection
python -m venv .venv
```

Activate the environment (**Windows PowerShell**):

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest tests/ -v
```

Run the traditional baseline experiments from the project root:

```powershell
python -m diplomacy.training.experiment_runner --root . --quick
```

Reproduce Phase 6 artifact-based analysis after restoring the required prediction and metric files:

```powershell
python -m diplomacy.analysis.phase6_analysis --root . --output artifacts/phase_6
python -m diplomacy.analysis.phase6_phase5_deep_dive --root . --output artifacts/phase_6
```

GPU training for the neural experiments was performed in **Google Colab**, using the notebooks in `notebooks/`. Production model and training code is shared under `src/diplomacy/`. The final verification recorded **71 passing automated tests**.

## Limitations

- The dataset contains only 12 games; the test split has just two games, and lie labels are imbalanced.
- BERT-based Phase 5 models did not identify positive examples at the original threshold; the precise cause is not conclusively established.
- Several architectures and training choices are research-inspired adaptations, not exact reimplementations of every original-paper detail.
- Research conclusions should not be interpreted as reliable real-world lie detection.

## Acknowledgements and Reference

This is an independent implementation inspired by **Peskov et al. (2020)**, [*It Takes Two to Lie: One to Lie, and One to Listen*](https://aclanthology.org/2020.acl-main.353/), ACL 2020. See the authors' [original research repository](https://github.com/DenisPeskoff/2020_acl_diplomacy) for their implementation. This project is separate from and not an official release of that work.

Further research and experimentation are planned as future work.
