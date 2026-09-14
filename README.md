# Document-Level KG-to-Text Generation

Research code for experiments on sentence- and document-level knowledge-graph-to-text generation, including Direct prompting, Chain-of-Thought (CoT), Fact-by-Fact generation (FBF), FactSpotter-Guided Consistency (FGC), QLoRA fine-tuning, and the GAP graph-aware baseline.


## Repository layout

```text
.
├── src/
│   ├── data/                 # dataset loaders and dataset-construction utilities
│   ├── evaluation/           # BLEU, BERTScore, BLEURT, AlignScore, FactSpotter
│   ├── models/               # local / OpenAI / OpenRouter model wrappers
│   ├── pipelines/            # Direct, CoT, FBF, QLoRA, GAP
│   ├── rerankers/            # FactSpotter reranking used by FGC
│   ├── run_experiment.py     # single-run entry point
│   ├── run_n_seeds.py        # multi-seed launcher
│   └── run_method_multiseed.py
├── data/                     # datasets
├── requirements.txt
└── .env.example
```

## Setup

Create an isolated Python environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Running experiments

Run commands from the repository root.

### Direct prompting

```bash
python -m src.run_experiment \
  --dataset wikidockg \
  --output_dir results/wikidockg/qwen9b-direct \
  --run_direct \
  --policy_model_name Qwen/Qwen3.5-9B
```

### Fact-by-Fact generation

First-stage fact realization only:

```bash
python -m src.run_experiment \
  --dataset wikidockg \
  --output_dir results/wikidockg/qwen9b-fbf-stage1 \
  --run_fbf \
  --fbf_first_stage_only \
  --policy_model_name Qwen/Qwen3.5-9B
```

Two-stage FBF with discourse naturalization:

```bash
python -m src.run_experiment \
  --dataset wikidockg \
  --output_dir results/wikidockg/qwen9b-fbf-stage2 \
  --run_fbf \
  --policy_model_name Qwen/Qwen3.5-9B
```

### FactSpotter-Guided Consistency (FGC)

FGC is implemented through candidate aggregation with `--aggregation factspotter`. For Direct prompting, this automatically generates five candidates and selects the candidate preferred by FactSpotter:

```bash
python -m src.run_experiment \
  --dataset wikidockg \
  --output_dir results/wikidockg/qwen9b-direct-fgc \
  --run_direct \
  --aggregation factspotter \
  --policy_model_name Qwen/Qwen3.5-9B
```

The same aggregation option can be used with CoT and fine-tuned generation.

### QLoRA fine-tuning

```bash
python -m src.run_experiment \
  --dataset wikidockg \
  --output_dir results/wikidockg/qwen9b-qlora \
  --run_finetune \
  --finetune_model_name Qwen/Qwen3.5-9B \
  --finetune_use_lora \
  --load_in_4bit \
  --finetune_epochs 1
```

### GAP baseline

```bash
python -m src.run_experiment \
  --dataset wikidockg \
  --output_dir results/wikidockg/gap \
  --run_gap \
  --gap_model_name facebook/bart-base
```


## Evaluation

We recommend using our Dockerized evaluation API for reproducing the metrics
reported in the paper. It provides a unified interface to BLEU, BERTScore,
BLEURT, AlignScore, and FactSpotter, while fixing the metric implementations,
model checkpoints, and dependencies used in our experiments.

The API repository, including Docker deployment instructions and client
examples, is available at:

https://github.com/tchewik/kg2text-eval

For reproducibility, we recommend the hosted/remote API when available.
The same service can be deployed locally with Docker for offline use.

```bash
python -m src.evaluation.eval_n_seeds_remote \
  --base_output_dir results/wikidockg/qwen9b-direct \
  --pred_relpath direct/test_predictions.jsonl \
  --api-url https://<HOST>/v1/evaluate/file
```

