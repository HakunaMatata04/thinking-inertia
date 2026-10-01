# Domain aggregation

`aggregate_mmlu_domain_metrics.py` joins QRel. scores with blinded EIR labels
and aggregates them by MMLU subject area. It expects the public normalized
MMLU input plus external, text-free score/label tables. Generated area and
subject aggregates go under the ignored local `outputs/domain/` directory; raw
records and judge labels are intentionally omitted.
