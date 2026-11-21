# Test strategy

## Layers
| Layer | File | Needs network/keys | Runs in CI | Purpose |
|---|---|---|---|---|
| Unit | `tests/test_chunkers.py` | no | yes | Chunk size, overlap, no lost words, invalid params |
| Contract (mocked) | `tests/test_rag_pipeline_offline.py` | no | yes | Prompt grounding, retrieval limits, LLM params, error propagation |
| LLM evaluation | `tests/test_faithfulness.py` | yes (Groq) | no (opt-in) | Are answers grounded in retrieved chunks (DeepEval FaithfulnessMetric) |

Run offline suite: `pip install -r requirements-dev.txt && pytest -q`
Run live eval: `RUN_LIVE_EVAL=1 python tests/test_faithfulness.py --threshold 0.6`

## Design decisions
- **Deterministic first.** Everything that can be tested without an LLM is, and it gates CI.
  Heavy dependencies (embedding model, ChromaDB, Groq, Langfuse) are mocked, so the suite
  runs in under a second.
- **LLM eval is a quality signal, not a gate.** An LLM judge is non-deterministic and costs
  money/quota, so it runs on demand and its score is tracked, not used to fail a push.
- **Golden dataset** (`GOLDEN_DATASET` in `test_faithfulness.py`): questions with categories,
  covering hardware, storage etc. Grow it when a bug is found (regression case first).

## Bugs found by testing
- `chunk_text` / `fixed_size_chunking` looped forever when `overlap >= chunk_size`; now raises `ValueError`.
- The windowing emitted a redundant tail chunk fully contained in the previous one; removed.
  (Re-run `ingest.py` to rebuild the index with the corrected chunking.)
- Hardcoded absolute paths broke the project on any other machine; now relative, overridable with `PDF_PATH` / `CHROMA_PATH`.

## Known limitations / next steps
- Judge model (Groq Llama) differs from generator family only by config; self-preference bias is possible.
- Faithfulness does not measure answer *relevance* or *retrieval recall*; add contextual-recall and answer-relevancy metrics.
- No negative/adversarial set yet (prompt injection, out-of-scope questions should return the fallback sentence).
- `chatbot.py` mixes web UI, retrieval and generation; splitting it would make each part directly testable.
