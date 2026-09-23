# Starter training reference

Follow the README commands from the repository root. `prepare` validates record schemas and exact split overlaps, then refuses to overwrite an existing workshop. `download` retrieves the pinned ModernBERT model. `baseline` measures the original; `train` saves a separately trained checkpoint; `test` verifies frozen hashes and preserves its final report.

The included data is intentionally small and synthetic. `--limit 8` checks the pipeline using eight decisions per split. It cannot establish useful performance. For a real task, use checked representative examples and separate train/dev/test files; pass their directory with `prepare --data /path/to/data --run workshops/my-task`. Use a new workspace, then omit `--limit` consistently from baseline, training and test.

A record needs an id, state text, and questions. Each question has three distinct candidate descriptions and a gold index of 0, 1 or 2. See the supplied JSONL examples. Keep related documents together and inspect labels manually. Never tune on the final test and call it unseen again.

To predict, create input.json with a state string and three candidate descriptions:

```json
{"state":"Cancellations receive hotel credit only.","candidates":["A full cash refund is available.","A full cash refund is not available.","Cash refund availability is not specified."]}
```

```bash
uv run --project apps/agency python tools/tutorial.py predict --run workshops/first-check --input input.json
```

Inspect the actual result and the saved reports. Tiny-data outputs are not reliable travel advice. The supplied model in the video, full local app and optional image service belong to the complete community edition. This starter does not need or include those checkpoints.


## Build your own version with us

Get the complete project code, training pipeline, model checkpoints, and exclusive training inside Early AI Adopters.

[Join Early AI Adopters](https://www.skool.com/earlyaidopters/about)
