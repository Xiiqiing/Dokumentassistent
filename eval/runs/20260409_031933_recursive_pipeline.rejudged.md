# RAGAS Evaluation — 20260409_031933

## Setup

- Generation LLM: `groq` / `qwen/qwen3-32b`
- Judge LLM:      `groq` / `llama-3.3-70b-versatile`
- Embeddings:     `local` / `paraphrase-multilingual-MiniLM-L12-v2`
- Reranker:       `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Samples:        33

## Aggregate Scores

| Config | Chunking | Router | top_k | faithfulness | answer_relevancy | llm_context_precision_with_reference | context_recall | answer_correctness | factual_correctness(mode=f1) |
|---|---|---|---|---|---|---|---|---|---|
| recursive_pipeline | recursive | pipeline | 5 | 0.7880 | 0.8659 | 0.7244 | 0.7879 | 0.4506 | 0.4009 |
