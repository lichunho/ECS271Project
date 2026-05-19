# Implementation plan

## 1. Data preparation
- Pull the six QA datasets (MuSiQue, HotpotQA, 2WikiMultiHopQA, SQuAD v1.1, NaturalQuestions, TriviaQA)
- Generate the non-retrieval set following Adaptive-RAG's procedure
- Apply oracle labels (`none` / `single` / `multi`) to every query — these are your training targets for the classifier and your ground truth for evaluation
- Split into train / val / test

## 2. Retrieval and corpus setup
- Set up the retriever Adaptive-RAG uses (the Wikipedia corpus + retriever model from their repo)
- Verify retrieval works end-to-end on a handful of queries before moving on

## 3. Stage 1 — BERT routing classifier
- Fine-tune BERT-base for 3-way classification on the labeled queries
- Hold out a val set for picking the checkpoint
- Sanity-check: per-class accuracy on `none` / `single` / `multi` so you can compare against the T5 numbers in the Adaptive-RAG paper (30 / 66 / 65%)

## 4. Answering LLM setup
- Get quantized Qwen 3 running on your hardware
- Wire up two call modes: a short "probe" generation (one sentence, return token logprobs) and a full answer generation
- Make sure the same model/prompt is used for both — otherwise the probe doesn't reflect the answerer's knowledge

## 5. Stage 2 — confidence probe
- Decide your confidence metric (mean logprob, min token prob, etc.) and threshold τ
- Implement the gate: if BERT predicts `multi`, run the probe; if confidence > τ, demote to `single`
- Tune τ on the validation set

## 6. Full pipeline integration
- Glue Stage 1 + Stage 2 + retrieval + final answer generation into one routing policy `f(q) → ŝ → answer`
- Each route (`none` / `single` / `multi`) needs its own execution path wired up

## 7. Baselines
- Always-no-retrieval, always-single, always-multi (these are cheap — just force the route)
- Standard Adaptive-RAG with their T5 classifier (use their released checkpoint if available, otherwise reproduce)

## 8. Evaluation
- Run your system and all baselines on the test set
- Compute: EM, F1, routing accuracy vs. oracle, per-query latency, token cost
- Break results out by dataset and by oracle class so you can see *where* the hybrid helps and where it doesn't

## 9. Analysis and writeup
- Confusion matrices for routing decisions (BERT alone vs. BERT + probe)
- Ablation: BERT only, probe only (FLARE-style), full hybrid
- Error analysis: pick examples where the probe correctly demoted a query, and examples where it shouldn't have

---

A couple of things worth knowing now rather than later:

- **Step 1 is the longest pole.** Generating the non-retrieval set and getting oracle labels right is fiddly and easy to underestimate. Start there.
- **Steps 2 and 4 can run in parallel** with steps 3 and 5 if you split work between teammates.
- **Step 7 (the standard Adaptive-RAG baseline) is non-trivial** — reproducing someone else's pipeline always takes longer than expected. If their code/checkpoints aren't usable as-is, consider whether you can scope it down (e.g., only compare on a subset of datasets).