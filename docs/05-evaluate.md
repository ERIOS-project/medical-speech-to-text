# 05 — Standalone evaluation

**TODO: `medstt evaluate` is not implemented.** The command exists but exits with an error. The evaluation protocol and report format still need to be defined in [TODO.md](../TODO.md).

For now, configure `train.data.eval` or `train.data.eval_split` in [`medstt.yaml`](04-train.md). WER/CER metrics computed during training are saved in `<workdir>/runs/<run_name>/eval_results.json`.

A standalone evaluation should compare the base model and the adapted model on held-out medical terms and general speech, with per-dataset and per-term results.
