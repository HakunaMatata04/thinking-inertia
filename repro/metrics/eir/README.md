# EIR judge and aggregation

This directory contains the public, text-free part of the Level 3 pipeline:
the blind rubric, corpus preparation, asynchronous judge client, and
all-output EIR aggregation. The judge receives only `(Q, T)` and returns one
exclusive category plus a supporting span. Raw questions, pre-answer text,
provider responses, and per-example labels are not included here.

The categories are defined in `judge_rubric.py` and the same rubric is used by
the GPT-5.5 and cross-family judge analyses. Generated aggregate files are
written to the ignored local `outputs/eir/` directory.

To run the pipeline on an external corpus:

```bash
python repro/metrics/eir/prepare_corpus.py \
  --input /path/to/records.jsonl \
  --output-dir outputs/eir

DMX_API_KEY=... python repro/metrics/eir/judge_pairs.py \
  --input outputs/eir/m2_corpus.jsonl \
  --output /tmp/gpt55_labels.jsonl

python repro/metrics/eir/aggregate.py \
  --corpus outputs/eir/m2_corpus.jsonl \
  --labels /tmp/gpt55_labels.jsonl \
  --output-dir outputs/eir
```
