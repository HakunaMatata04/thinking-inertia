# QRel. scoring

This scorer computes instruction-aware question--pre-answer relevance (QRel.)
with Qwen3-Embedding-4B. It accepts the controlled-experiment `records.jsonl`
files as an external input; raw records are not distributed in this repository.

- Query: the complete question `Q`, encoded with the instruction `Given a
  question, retrieve pre-answer text that contains reasoning relevant to
  answering the question`.
- Candidate: the extracted pre-answer text `T`.
- Score: the dot product of L2-normalized embeddings; empty `T` receives zero.

Example:

```bash
python -m thinking_inertia.metrics.qrel.score_controlled_qrel \
  --run-root /path/to/controlled-experiments \
  --model-path Qwen/Qwen3-Embedding-4B \
  --device cuda:0 \
  --batch-size 4 \
  --output-dir outputs/qrel
```

The scorer writes per-example `item_qrel.csv` and grouped
`aggregate_qrel.csv` under the ignored local output directory. The source
questions and pre-answer text remain external inputs. Paper-specific full-corpus
aggregation and figure generation are intentionally not part of this public
source release.
