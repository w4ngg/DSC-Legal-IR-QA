# Retrieval recall analysis

`evaluate_retrieval_recall.py` evaluates retrieval-stage document recall from
the regular and deep diagnostics emitted by `legal_ir search`.

The lane cutoff is always a **chunk cutoff**. The script truncates each
channel's `chunk_hits` first and only then deduplicates/aggregates the retained
chunks to documents. It never truncates the precomputed `document_hits` list.

## BM25 + dense

```bash
python analysis/evaluate_retrieval_recall.py \
  --gold runs/version1/val.json \
  --deep-diagnostics runs/new_index/deep_diagnostics.json \
  --diagnostics runs/new_index/diagnostics.json \
  --bm25-top-k 150 \
  --dense-top-k 100 \
  --fusion-cutoffs 5,10,20,30,50 \
  --final-cutoffs 1,3,5 \
  --output runs/new_index/retrieval_recall_summary.json \
  --per-query-output runs/new_index/retrieval_recall_per_query.json \
  --strict-query-ids
```

The retrieval report contains separate BM25 and dense metrics plus
`oracle_union`, the deduplicated union of their document sets.

## Optional HyDE lane

Add `--hyde-top-k` to include HyDE in both its own stage and the oracle union:

```bash
python analysis/evaluate_retrieval_recall.py \
  --gold runs/version1/val.json \
  --deep-diagnostics runs/new_index/deep_diagnostics.json \
  --bm25-top-k 150 \
  --dense-top-k 100 \
  --hyde-top-k 100 \
  --output runs/new_index/retrieval_recall_with_hyde.json \
  --strict-query-ids
```

Omitting `--hyde-top-k` ignores a logged HyDE channel and makes
`oracle_union` mean BM25 + dense only.

`--diagnostics` is optional. When present, the script additionally evaluates
ordered `fused_candidates` and final `results` at the requested document
cutoffs. When the cutoff flags are omitted, it uses the fusion/final depths
recorded in `deep_diagnostics.pipeline_config`.

Headline metrics are macro recall, micro recall, Hit rate and FullHit rate.
Candidate lists may contain more than five documents, so this tool deliberately
does not label candidate-pool precision as the official Task 1 precision.

## Count long-chunk reranker candidates over a BM25/dense grid

`count_long_chunk_candidate_grid.py` truncates `chunk_hits` independently in
the BM25 and dense lanes, unions/deduplicates the resulting short chunk IDs,
maps them through `short_to_long.jsonl`, and finally deduplicates the long chunk
IDs. By default it evaluates the complete 7 x 7 grid whose lane cutoffs are
`100,150,200,250,300,350,400`.

The deep diagnostics must therefore have been generated with both BM25 and
dense `top_k_chunks >= 400`. The mapping may live in a mounted Kaggle Dataset;
the script streams that JSONL once and retains only mappings referenced by the
diagnostics.

```bash
python analysis/count_long_chunk_candidate_grid.py \
  --deep-diagnostics runs/new_index/deep_diagnostics.json \
  --short-to-long /path/to/dual_v1/short_to_long.jsonl
```

The default `--mapping-mode all` uses every ID in `long_chunk_ids`. Use
`--mapping-mode primary` only for a separate primary-long ablation. Without
`--output` or `--per-query-output`, the script writes no artifact and prints
three 7 x 7 matrices: mean long chunks/query, total query-long pairs, and P95
long chunks/query.

Optional JSON outputs:

```bash
python analysis/count_long_chunk_candidate_grid.py \
  --deep-diagnostics runs/new_index/deep_diagnostics.json \
  --short-to-long /path/to/dual_v1/short_to_long.jsonl \
  --output runs/new_index/long_candidate_grid_summary.json \
  --per-query-output runs/new_index/long_candidate_grid_per_query.json
```

## Tests

```bash
python -m unittest discover -s analysis -p 'test_*.py' -v
```
