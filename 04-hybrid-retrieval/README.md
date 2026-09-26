# 04 — Hybrid Retrieval for RAG

The retrieval half of a RAG system, built rather than imported: lexical search, dense search, and a reranker, combined into one pipeline over 50,000 Polish Wikipedia passages.

## The pipeline

```
query
  ├─ BM25              lexical  — exact term overlap, Polish stop words removed
  └─ bi-encoder        dense    — query and passages embedded independently
        │
        ▼
   union of both candidate pools
        │
        ▼
   cross-encoder       reranking — query and passage scored jointly
        │
        ▼
   top N passages → prompt with forced [n] citations → Llama-PLLuM-8B-instruct
```

## Why each stage is there

**BM25 and the bi-encoder fail differently.** Lexical search misses paraphrase; dense search misses rare proper nouns and exact identifiers it never saw in training. Taking the union of both candidate sets means a passage only has to be found by one of them.

**The cross-encoder is accurate but quadratic.** Scoring the query jointly with a passage is far better than comparing two independent embeddings, and far too expensive to run over 50k passages. So it never sees the corpus — only the few dozen candidates the first two stages produced. That ordering is the whole design.

**Chunking with overlap.** Documents are split into 200-word passages with a 40-word overlap, so a sentence that falls across a boundary still appears whole in one of the two neighbouring chunks.

**Grounding is enforced in the prompt**, which instructs the model to answer only from the supplied documents and to cite them as `[document_number]`.

## Models

`multi-qa-mpnet-base-dot-v1` (bi-encoder), `cross-encoder/ms-marco-MiniLM-L6-v2` (reranker), `CYFRAGOVPL/Llama-PLLuM-8B-instruct` (generator), corpus `chrisociepa/wikipedia-pl-20230401`.

## Run

Colab with a GPU; the corpus is streamed from Hugging Face, so no local data is needed.
