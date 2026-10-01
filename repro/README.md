# Reproducibility utilities

The public source release keeps the final metric implementations separate from
the model-serving runner:

- `metrics/qrel/` computes instruction-aware QRel. with Qwen3-Embedding-4B;
- `metrics/eir/` prepares `(Q, T)` pairs, applies the blinded rubric, and
  aggregates all-output EIR;
- `metrics/domain/` joins QRel. and EIR by MMLU area and subject;
- `metrics/human_validation/` documents the blinded annotation rubric without
  distributing response text or reviewer files.

The raw response corpus is intentionally an external input. Generated files
belong under `outputs/`, which is ignored by Git.
