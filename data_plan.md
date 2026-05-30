# Data sourcing & annotation plan

Extends `proposal.md` steps 1 ("Data preparation") with the concrete file paths,
commands, and contracts the team will follow. Decisions in this doc supersede the
sketch in `proposal.md` where they conflict.

All citations point at [`starsuzi/Adaptive-RAG`](https://github.com/starsuzi/Adaptive-RAG).

## Locked-in decisions

| Decision | Choice |
|---|---|
| Fidelity | **Reproduce Adaptive-RAG's pipeline end-to-end.** Don't reuse their released labels for our final data; do download them as a sanity baseline. |
| Scale | **Paper scale: 500/dataset × 2 splits = 6,000 questions × ~3 strategies ≈ 25–30k LLM calls.** (The phrase "tens of thousands" in `proposal.md` was an overestimate — Adaptive-RAG's `sample_size = 500` is hard-coded in `subsample_dataset_and_remap_paras.py`.) |
| Labeller / answerer LLM | **Parameterize.** The annotator takes `--model`. Cache keys include `(question_hash, strategy, model_id, prompt_set)` so swapping models invalidates only that model's slice. **Same model is used for labelling and Stage-2 probe at inference** — required for the probe to reflect the answerer's calibration. |
| BM25 retriever | **Pyserini's prebuilt `wikipedia-dpr-100w` index** for NQ/Trivia/SQuAD. No Elasticsearch. Build per-dataset BM25 indices for HotpotQA / 2Wiki / MuSiQue with `pyserini index` over their bundled corpora. Slight scoring divergence from paper's ES BM25 — documented as a deliberate trade. |
| Prompt mix | **Follow paper:** direct-QA for NQ/Trivia/SQuAD, CoT for HotpotQA/2Wiki/MuSiQue. Use Adaptive-RAG's `prompts/{dataset}/*_flan_t5.txt` files verbatim as completion-style few-shot (no system prompt). |

## On-disk layout

```
ECS271Project/
├── data/
│   ├── raw/                           # raw downloads (verbatim from sources)
│   │   ├── hotpotqa/{hotpot_train_v1.1.json, hotpot_dev_distractor_v1.json}
│   │   ├── 2wikimultihopqa/{train,dev,test}.json + id_aliases.json
│   │   ├── musique/musique_ans_v1.0_{train,dev,test}.jsonl
│   │   │           + dev_test_singlehop_questions_v1.0.json   # for overlap filter
│   │   ├── nq/biencoder-nq-{train,dev}.json                   # DPR-curated, NOT raw NQ
│   │   ├── trivia/biencoder-trivia-{train,dev}.json
│   │   └── squad/biencoder-squad1-{train,dev}.json
│   ├── processed/                     # unified JSONL schema, per-dataset
│   │   └── {dataset}/{train,dev}.jsonl
│   ├── eval_500/{dataset}.jsonl       # 500/dataset × 6 = 3,000 — our eval ("paper test")
│   ├── train_500/{dataset}.jsonl      # disjoint 500/dataset × 6 = 3,000 — classifier train pool
│   ├── predictions/                   # answerer LLM outputs per strategy
│   │   └── {split}/{strategy}_{model_id}_{dataset}/correct_qids.json
│   ├── labeled/
│   │   ├── classifier_train.jsonl     # silver + binary, ready for RoBERTa fine-tune
│   │   ├── classifier_valid.jsonl
│   │   └── eval_with_oracle_labels.jsonl
│   ├── reference/                     # download once for sanity-diff against our re-derived data
│   │   ├── processed_data_from_repo/
│   │   ├── predictions_from_repo/
│   │   └── classifier_data_from_repo/
│   └── indices/pyserini/              # BM25 indices (wiki = prebuilt, others = built locally)
├── scripts/
│   ├── download_data.py               # see §1
│   └── annotate.py                    # see §2
└── src/annotate_lib/                  # supporting modules for annotate.py
    ├── normalise.py                   # SQuAD normalize_answer (lower / strip punct / strip articles)
    ├── prompts.py                     # load Adaptive-RAG prompts/{dataset}/*.txt
    ├── extract.py                     # ".* answer is (.*)" regex
    ├── strategies.py                  # no_retrieval / single_step / multi_step (IRCoT)
    ├── retrieval.py                   # Pyserini BM25 client
    └── cache.py                       # SQLite WAL cache for LLM calls
```

## Unified row schema (the contract between phases)

Every JSONL line in `data/processed/`, `data/eval_500/`, `data/train_500/` uses **Adaptive-RAG's native schema** (preserved verbatim by our port of `process_*.py` for fidelity):

```jsonc
{
  "dataset": "musique",
  "question_id": "single_nq_dev_42",
  "question_text": "Who wrote the score for Star Wars?",
  "answers_objects": [{
    "number": "",
    "date": {"day": "", "month": "", "year": ""},
    "spans": ["John Williams"]              // gold answers (multiple acceptable)
  }],
  "contexts": [{
    "idx": 0,
    "title": "John Williams",
    "paragraph_text": "...",
    "is_supporting": true                   // gold-supporting flag
  }, ...],
  // multi-hop datasets also carry:
  "reasoning_steps": ["q1 >>>> a1", ...],   // musique / 2wiki
  "level": "hard", "type": "comparison"     // hotpotqa only
}
```

The annotation step adds `oracle_label`, `label_source`, `attempts`, and the labeller metadata to each row — see the labeled-row schema below. (Earlier drafts of this plan described an `id`/`question`/`gold_answers`-shaped schema; we keep the Adaptive-RAG shape on disk and let downstream code project the few fields it needs.)

After annotation (`data/labeled/eval_with_oracle_labels.jsonl` and `classifier_{train,valid}.jsonl`) — original Adaptive-RAG fields preserved, plus:

```jsonc
{
  // ...all of the processed-schema fields, plus:
  "oracle_label": "no_retrieval",                // no_retrieval | single_step | multi_step
  "label_source": "silver",                      // silver | binary_fallback
  "attempts": {
    "no_retrieval": {"pred_raw": "...", "pred_extracted": "...",
                     "em": 1, "f1": 1.0, "ok": true, "latency_s": 0.8},
    "single_step":  {"pred_raw": "...", "pred_extracted": "...",
                     "em": 0, "f1": 0.6, "ok": false, "latency_s": 2.1,
                     "context_doc_ids": ["...", "..."]},
    "multi_step":   {"pred_raw": "...", "pred_extracted": "...",
                     "em": 0, "f1": 0.4, "ok": false, "latency_s": 6.3,
                     "n_hops": 2, "context_doc_ids": [...]}
  },
  "labeller_model_id": "qwen3:7b-instruct-q4_K_M",  // whatever --model was
  "labeller_prompt_set": "flan_t5_v1",
  "labeller_commit": "<git-sha>"
}
```

---

## §1 — `scripts/download_data.py` (sourcing)

Step-by-step. Don't parallelise within a step until the prior step's checksum/diff passes.

| # | Step | Source | Output | Disk | ~Time |
|---|---|---|---|---|---|
| 1 | Make `data/raw/{nq,trivia,squad,wiki,hotpotqa,2wikimultihopqa,musique}` | — | empty tree | 0 | <1s |
| 2 | HotpotQA train + dev | `curtis.ml.cmu.edu/datasets/hotpot/hotpot_{train_v1.1,dev_distractor_v1}.json` | `raw/hotpotqa/*.json` | ~600 MB | 1–2 min |
| 3 | 2Wiki | Dropbox `data_ids.zip` ([`download/raw_data.sh:13`](https://github.com/starsuzi/Adaptive-RAG/blob/main/download/raw_data.sh)) | `raw/2wikimultihopqa/*.json + id_aliases.json` | ~500 MB | 1–2 min |
| 4 | MuSiQue | gdown `1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h` (Ans variant only) | `raw/musique/musique_ans_v1.0_*.jsonl + dev_test_singlehop_questions_v1.0.json` | ~600 MB | 2–5 min |
| 5 | DPR files for NQ / Trivia / SQuAD | `dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-{nq,trivia,squad1}-{train,dev}.json.gz` | `raw/{nq,trivia,squad}/biencoder-*.json` | ~3 GB | 2–5 min |
| 6 | Sanity baseline tarballs | `github.com/starsuzi/Adaptive-RAG/raw/main/{processed_data,predictions,data}.tar.gz` | `data/reference/*` | ~250 MB extracted | <1 min |
| 7 | Run six `processing_scripts/process_*.py` (port to our repo; pure Python, no deps beyond stdlib + tqdm) | `raw/*` | `processed/{dataset}/{train,dev}.jsonl` | ~3 GB | 5–15 min |
| 8 | Run `subsample_dataset_and_remap_paras.py {dataset} test 500` ×6 (seed `13370`) | `processed/{dataset}/dev.jsonl` | `processed/{dataset}/test_subsampled.jsonl` → `data/eval_500/{dataset}.jsonl` | ~25 MB | <1 min |
| 9 | Run `subsample_dataset_and_remap_paras.py {dataset} dev_diff_size 500` ×6 (disjoint from step 8) | `processed/{dataset}/dev.jsonl` | `processed/{dataset}/dev_500_subsampled.jsonl` → `data/train_500/{dataset}.jsonl` | ~25 MB | <1 min |
| 10 | Diff our `data/eval_500/*.jsonl` qids against `data/reference/processed_data_from_repo/{dataset}/test_subsampled.jsonl` (**all six datasets — the repo tarball ships all of them, not just multi-hop**) | both | report | 0 | <30 s |
| 11 | **Hard checkpoint**: step 10 must report identical sets. Same seed alone is *not* sufficient — see §1a. | — | — | — | — |

### §1a — Reproducibility: project the reference qids onto our processed dev

Adaptive-RAG's reference `test_subsampled.jsonl` / `dev_500_subsampled.jsonl` are not reproducible from `random.seed(13370) + random.sample(dev, 500)` alone — their sampling-time RNG state differs from a fresh seed in ways that can't be reverse-engineered (we tried; for single-hop datasets no `N` near 6515 gives matching output). The IRCoT seed file worked for multi-hop but **not** for single-hop, where IRCoT doesn't ship an avoid file.

**Default strategy (used by `scripts/download_data.py` unless `--no-reference-qids` is passed):** read the reference qid set from `data/reference/processed_data_from_repo/{dataset}/{test,dev_500}_subsampled.jsonl` (downloaded once in step 6) and **project it onto our processed dev** — keeping rows in reference order. The reference is the canonical paper split, so this is byte-identical to what the paper used. All six datasets pass the step-10/11 diff this way.

The orchestrator validates that every reference qid is present in our processed dev; if not, it halts (would indicate our `process_*.py` port dropped rows the upstream kept).

**Legacy random-sample path** (`--no-reference-qids`): for multi-hop datasets, seeds IRCoT's `dev_subsampled.jsonl` from Google Drive `1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC` (the ~1.7 GB `processed_data.zip`); for single-hop datasets, sampling diverges from the reference and step 11 fails. Mainly useful if a future contributor wants to regenerate splits without depending on the reference tarball.
| 12 | Pull Pyserini's prebuilt index for single-hop datasets | `python -m pyserini.search.lucene.SimpleSearcher --download wikipedia-dpr-100w` | `~/.cache/pyserini/...` | ~20 GB | 10–20 min one-time |
| 13 | Build BM25 over per-dataset corpora (HotpotQA / 2Wiki / MuSiQue) | `pyserini index --collection ...` | `data/indices/pyserini/{dataset}/` | ~10 GB combined | HotpotQA ~30 min, others <10 min |
| 14 | Smoke-test retrieval: a known query per dataset returns a known passage | — | report | 0 | <1 min |

**Total disk:** ~39 GB (raw + processed + reference tarballs + IRCoT seed cache ≈ 1.7 GB). **Total wall-clock:** ~1 hour for download (steps 1–11) + ~30–60 min for retrieval setup (steps 12–14).

**Paragraph remap (post-step-14):** once retrieval indices are built, you can opt into the upstream "rewrite context titles/paragraph_text from the BM25 corpus" pass by re-running step 8 with `--remap-paragraphs --force --step 8` (or via `python -m src.processing.subsample {dataset} test --input-dir data/processed/{dataset} --remap-paragraphs`). Default off — the qid set is identical either way, so step-10/11 diffs pass either way. Recommend turning it on so the eval-time BM25 lookup keys line up with the saved gold contexts.

### Per-dataset version & schema notes

- **NQ / Trivia / SQuAD use DPR's curated subsets**, not raw HF datasets. Mention in writeup.
- The MuSiQue `dev_test_singlehop_questions_v1.0.json` is used to filter NQ-train overlap (`get_overlapped_qid()` in `preprocess_utils.py`). Easy to miss.
- Paper's "test" = **original dev split** (true test splits are hidden/unavailable for NQ/Trivia/HotpotQA-distractor).
- The repo's `processed_data.tar.gz` only contains the **4 multi-hop datasets** (+ IIRC). For NQ/Trivia/SQuAD we *must* run our port of `process_*.py`.

### Risks
- HotpotQA Stanford host can be slow/down → mirror from IRCoT's copies.
- DPR `biencoder-*` files use `positive_ctxs` / `negative_ctxs` / `hard_negative_ctxs` — `process_{nq,trivia,squad}.py` keeps "up to 5 negatives and 5 hard negatives per instance" + all positives.
- Hardcoded paths in Adaptive-RAG (`classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/...`) — mirror naming or refactor our port.

---

## §2 — `scripts/annotate.py` (annotation)

### The procedure (reproducing Adaptive-RAG exactly)

For each question in `data/{eval_500,train_500}/{dataset}.jsonl`, run three strategies. **All three always run** (we want full attempt records for analysis):

| Strategy | Label | Retrieval | Prompt | Decoding |
|---|---|---|---|---|
| `no_retrieval` | A | none | `prompts/{dataset}/no_context_{direct,cot}_qa_flan_t5.txt` | temperature=0, max_tokens=200 |
| `single_step` | B | BM25 top-15, take first 15 unique passages | `prompts/{dataset}/gold_with_1_distractors_context_{direct,cot}_qa_flan_t5.txt` | temperature=0, max_tokens=200 |
| `multi_step` | C | IRCoT loop: BM25 top-6 per step, max 4 iterations | same CoT template, re-prompted with growing context | temperature=0, max_tokens=200 |

Prompt convention per dataset:
- **Direct-QA**: NQ, TriviaQA, SQuAD
- **CoT**: HotpotQA, 2WikiMultiHopQA, MuSiQue

### Correctness gate (per-strategy)

EM = 1 after SQuAD `normalize_answer`:

```python
def normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s
```

For CoT outputs, first extract with regex `.* answer is (.*)` and strip a trailing period. Compare against every gold alias (`max_over_aliases`). F1 logged but **not used** as the gate.

### Label assignment (cheapest-wins, verified from `preprocess_utils.label_complexity`)

```python
attempts = {"no_retrieval": ..., "single_step": ..., "multi_step": ...}
ok = {s: a["ok"] for s, a in attempts.items()}

if ok["multi_step"]:    label = "multi_step"
if ok["single_step"]:   label = "single_step"   # overwrites multi
if ok["no_retrieval"]:  label = "no_retrieval"  # overwrites single
else:                   label = binary_fallback(source_dataset)

# binary fallback: MuSiQue/HotpotQA/2Wiki → multi_step; NQ/Trivia/SQuAD → single_step
# (NO inductive-bias-A exists — no_retrieval can only come from silver run.)
```

The three `if`s (not `elif`s) in the original code mean A wins if any of the strategies succeeded with A, even if B and C also succeeded. This is the "demote to simplest sufficient strategy" rule.

### Caching & resumability

- SQLite (WAL mode) at `data/.llm_cache.sqlite`.
- Key: `(question_hash, strategy, model_id, prompt_set)` where `question_hash = sha1(question.strip().lower())`.
- Every `_complete()` call goes through the cache. Crashes lose only the in-flight call.
- Output JSONL is append-only; final atomic dedup pass at end.
- `--dry-run` prints cache hit rate + estimated remaining wall-clock before committing.

### CLI

```
python scripts/annotate.py \
    --dataset hotpotqa \
    --split train_500 \
    --strategies no_retrieval,single_step,multi_step \
    --concurrency 4 \
    --max-questions 500 \
    --model qwen3:7b-instruct-q4_K_M \
    --bm25-endpoint http://localhost:8000
```

### Throughput estimate

- ~3–5 s/call direct QA, ~5–8 s/call CoT (~1.5k-token few-shot prefix).
- 6,000 questions × 3 strategies × ~1.5× (IRCoT loop avg) ≈ 25–30k calls.
- Local 7B-Q4, 4-way async (Ollama's `OLLAMA_NUM_PARALLEL=4`): **~15–20 hours**.
- Plan: **two overnight runs**, cache resumes the second.

### Quality control (do before training the classifier)

1. **Per-class histogram per dataset.** Expectation: SQuAD/NQ/Trivia tilt `no_retrieval`/`single_step`; HotpotQA/2Wiki tilt `multi_step`; MuSiQue heavily `multi_step`.
2. **Joint pass rate** (any strategy correct). If <50% on a single-hop dataset, the LLM is too weak — pause and reconsider model choice.
3. **`no_retrieval` ⊂ `single_step`** check: when closed-book passes, single-hop should also pass ~90%+. Lower → retrieval noise.
4. **50-example human spot-check**, stratified by `(source_dataset, oracle_label)`. Accept if ≥85% plausible.
5. **Prompt-leakage filter**: drop any qid that appears in `# METADATA: {"qid": "..."}` headers of `prompts/{dataset}/*.txt` (a small set, ~600 qids across all prompts).

### Risk register

| Risk | Trigger | Mitigation |
|---|---|---|
| `no_retrieval` class collapses (<5%) | Local 7B too weak on TriviaQA/NQ closed-book | Try a stronger model; otherwise lower the gate to F1≥0.6 *for no_retrieval only* and document deviation. **Cannot** silently break the model coupling — that would invalidate Stage-2 probe. |
| Wall-clock blows past 40 hr | Slower-than-expected throughput | Drop `multi_step` for NQ/Trivia/SQuAD (almost never wins); cap IRCoT at 2 iterations. |
| BM25 not ready before annotation | Phase 1 step 12/13 slips | Gold-docs fallback for `single_step`/`multi_step` on train labelling only. Re-label once BM25 is up. Never use this for eval. |
| Drift between labeller and final answerer | Someone swaps `LLM_MODEL` between labelling and inference | Annotator records `labeller_model_id` in every output row; classifier training script asserts uniform; inference script asserts match. |
| Few-shot prompt leakage to eval | Some `# METADATA` qids in prompt files overlap our eval_500 | Drop matching qids from eval (small set). |

---

## §3 — Open work for the next phase

Once annotation is done, the next agents pick up:
- **Classifier agent**: fine-tunes RoBERTa-large in `src/classifier.py` on `data/labeled/classifier_{train,valid}.jsonl`.
- **Pipeline agent**: integrates Stage-1 (RoBERTa) + Stage-2 (confidence probe, built on `src/annotate_lib/llm_adapter.py`) + retriever + per-route execution. Uses `data/labeled/eval_with_oracle_labels.jsonl` for eval.

## Key source files in Adaptive-RAG to reference

- [`download/raw_data.sh`](https://github.com/starsuzi/Adaptive-RAG/blob/main/download/raw_data.sh) — multi-hop raw downloads
- [`processing_scripts/process_*.py`](https://github.com/starsuzi/Adaptive-RAG/tree/main/processing_scripts) — six dataset normalisers
- [`processing_scripts/subsample_dataset_and_remap_paras.py`](https://github.com/starsuzi/Adaptive-RAG/blob/main/processing_scripts/subsample_dataset_and_remap_paras.py) — seed 13370, sample_size = 500
- [`classifier/preprocess/preprocess_silver_train.py`](https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/preprocess/preprocess_silver_train.py) — silver-label producer
- [`classifier/preprocess/preprocess_utils.py`](https://github.com/starsuzi/Adaptive-RAG/blob/main/classifier/preprocess/preprocess_utils.py) — `label_complexity`, `save_inductive_bias_*`, `get_overlapped_qid`, `concat_and_save_binary_silver`
- [`evaluate.py`](https://github.com/starsuzi/Adaptive-RAG/blob/main/evaluate.py) — `normalize_answer` + EM gate
- [`commaqa/inference/ircot.py`](https://github.com/starsuzi/Adaptive-RAG/blob/main/commaqa/inference/ircot.py) — `AnswerExtractor.query` and the `.* answer is (.*)` regex
- [`base_configs/ircot_qa_flan_t5_xl_hotpotqa.jsonnet`](https://github.com/starsuzi/Adaptive-RAG/blob/main/base_configs/ircot_qa_flan_t5_xl_hotpotqa.jsonnet) — BM25 config, max_length=200
- [`prompts/{dataset}/*.txt`](https://github.com/starsuzi/Adaptive-RAG/tree/main/prompts) — verbatim few-shot prompt files
