# ECS271Project

Adaptive RAG pipeline (in the style of Jeong et al. 2024, *"Adaptive-RAG"*). A small classifier routes incoming queries to one of three retrieval strategies:

- **A** — no retrieval (LLM answers directly)
- **B** — single-step retrieval (one-hop)
- **C** — multi-step retrieval (multi-hop)

This project uses encoder-only classifiers such as DeBERTa-v3-large or
RoBERTa-large as compact routing models. Adaptive-RAG's released classifier is
trained as a T5 seq2seq model that generates `A` / `B` / `C`; this repo
supports both the compact encoder classifier and a comparable T5 seq2seq
classifier.

## Setup (Windows + RTX 50-series / Blackwell)

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install PyTorch with CUDA 12.8 wheels (required for Blackwell sm_120):
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
   If the stable wheel fails on Python 3.13, fall back to the nightly index:
   ```powershell
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. Install the rest:
   ```powershell
   pip install -r requirements.txt
   ```

4. Verify CUDA visibility:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Expected: `True NVIDIA GeForce RTX 5070 Ti` (or whatever GPU is present).

5. Prefetch model weights:
   ```powershell
   python -m scripts.download_model
   ```

6. Smoke tests:
   ```powershell
   pytest tests/ -v
   ```

To change where weights are cached, copy `.env.example` to `.env` and set `HF_HOME`, or set the env var directly.

## Classifier Training

Once `data/labeled/classifier_train.jsonl` and
`data/labeled/classifier_valid.jsonl` exist, fine-tune a simple silver-only
router:

```powershell
python -m scripts.train_classifier `
  --train-file data/labeled/classifier_train.jsonl `
  --validation-file data/labeled/classifier_valid.jsonl `
  --model-name microsoft/deberta-v3-large `
  --output-dir outputs/classifier/deberta-v3-large-silver `
  --epochs 15 `
  --batch-size 8 `
  --eval-batch-size 32 `
  --gradient-accumulation-steps 4 `
  --learning-rate 3e-5 `
  --max-length 384 `
  --fp16
```

The trainer maps `oracle_label` to the same route IDs used by
`src.config.LABEL_MAP`:

- `no_retrieval` -> `0`
- `single_step` -> `1`
- `multi_step` -> `2`

It saves the best validation checkpoint, `best_metrics.json`, `history.json`,
and `label_map.json` under the output directory.

To mirror Adaptive-RAG's `binary_silver/train.json` setup more closely, first
build an augmented train file from Adaptive-RAG's binary-prior training rows.
If the sibling repo is available at `../Adaptive-RAG`, extract its binary file:

```powershell
mkdir data/reference
tar -xOzf ../Adaptive-RAG/data.tar.gz `
  ./classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/binary/total_data_train.json `
  > data/reference/adaptive_binary_total_data_train.json
```

Then combine those binary rows with this repo's true silver rows:

```powershell
python -m scripts.build_binary_silver_classifier_data `
  --adaptive-binary-json data/reference/adaptive_binary_total_data_train.json `
  --silver-train-file data/labeled/classifier_train.jsonl `
  --output-file data/labeled/classifier_train_binary_silver.jsonl
```

Then train DeBERTa-v3-large with:

```powershell
python -m scripts.train_classifier `
  --train-file data/labeled/classifier_train_binary_silver.jsonl `
  --validation-file data/labeled/classifier_valid_silver.jsonl `
  --model-name microsoft/deberta-v3-large `
  --output-dir outputs/classifier/deberta-v3-large-binary-silver `
  --fp16
```

To run an Adaptive-RAG-style T5 classifier on the same binary+silver/silver
split, use:

```powershell
python -m scripts.train_t5_classifier `
  --train-file data/labeled/classifier_train_binary_silver.jsonl `
  --validation-file data/labeled/classifier_valid_silver.jsonl `
  --model-name t5-large `
  --output-dir outputs/classifier/t5-large-binary-silver `
  --epochs 15 `
  --batch-size 8 `
  --eval-batch-size 32 `
  --gradient-accumulation-steps 4 `
  --learning-rate 3e-5 `
  --max-length 384 `
  --bf16 `
  --no-epoch-checkpoints
```

This script trains T5 to emit `A`, `B`, or `C` and evaluates by comparing the
first generated-token scores for those three options, matching the original
Adaptive-RAG classifier setup more closely than an encoder classification head.

To precompute initial router choices for the 500-example test/eval split, run
one route-prediction job per saved classifier:

```powershell
python -m scripts.predict_routes `
  --data-dir data/eval_500 `
  --model-kind encoder `
  --model-path outputs/classifier/deberta-v3-base-binary-silver `
  --classifier-name deberta-v3-base-binary-silver `
  --output-dir outputs/routes `
  --batch-size 32 `
  --max-length 384 `
  --include-answers
```

This writes `outputs/routes/{classifier-name}.routes.jsonl` and
`outputs/routes/{classifier-name}.summary.json`. Point `--output-dir` at a
Google Drive path in Colab to persist the files.

The trainer saves the best model at `--output-dir` and writes full resumable
epoch checkpoints under `--output-dir/checkpoint_epoch_N`. To resume after an
interruption:

```powershell
python -m scripts.train_classifier `
  --train-file data/labeled/classifier_train_binary_silver.jsonl `
  --validation-file data/labeled/classifier_valid_silver.jsonl `
  --output-dir outputs/classifier/deberta-v3-large-binary-silver `
  --resume-from-checkpoint outputs/classifier/deberta-v3-large-binary-silver/checkpoint_epoch_8 `
  --fp16
```

The binary+silver builder requires `data/processed/{dataset}/train.jsonl` for
all six datasets only if `--adaptive-binary-json` is not supplied. The currently
committed `data/train_500` split is not enough to add Adaptive-RAG-style
binary-prior examples, because those rows are already the silver training
questions.

## LLM (Ollama)

The answering LLM is served by [Ollama](https://ollama.com/), reached at the `LLM_BASE_URL` you set in `.env` (see `.env.example`) — which may point at a remote host. Ollama is used (rather than LM Studio) because as of LM Studio 0.3.x the OpenAI-compat layer returns `null` for per-token logprobs — which the Step 5 confidence probe depends on.

1. Install Ollama from [ollama.com/download](https://ollama.com/download). The installer registers a background service; no further start step is needed.
2. Pull a model:
   ```powershell
   ollama pull gemma4:26b   # ~16 GB; or any chat model you prefer
   ```
3. Point `LLM_MODEL` at whatever you pulled (defaults to `gemma4:26b`):
   ```powershell
   $env:LLM_MODEL = "gemma4:26b"
   ```
4. Verify the model is available:
   ```powershell
   ollama list
   ```
   `LLM_MODEL` should appear in the list (run this on whichever host serves Ollama).

The LLM client ([src/annotate_lib/llm_adapter.py](src/annotate_lib/llm_adapter.py)) talks Ollama's native `/api/chat` endpoint with a single bare user-role message (no system prompt) and `think: false`, so the completion-style Adaptive-RAG few-shot prompts are continued literally.

`LLM_BASE_URL` / `LLM_MODEL` are set in `.env` (see `.env.example`). The client assumes an Ollama backend; pointing it at a non-Ollama server would require a new branch in `llm_adapter.py`.

## Roadmap

- [x] Local RoBERTa classifier loads + runs inference on GPU
- [x] Training data: synthesize A/B/C labels from SQuAD / HotpotQA / MuSiQue
- [x] Fine-tune the classifier
- [x] Retriever (BM25)
- [x] LLM client (Ollama native `/api/chat`)
- [ ] Confidence-gated routing
- [ ] End-to-end eval harness
