# Frozen V2 challenger: final evidence

Local agreement **95.28%** versus **98.61%** for Jev. The superiority gate failed; the challenger was not promoted.

360 scenarios, 1,440 decisions per engine, no failed decisions. Paired accuracy difference -3.33 percentage points; scenario-bootstrap 95% interval [-4.38, -2.29].

Local macro F1 0.9528; 393 accepted decisions, zero observed accepted errors; correct-meets recall 83.44%. Quality and coverage floors passed. Zero observed errors does not establish zero population risk.

## Deployment and replication

Conditional replication not started because the first superiority gate failed.
The localhost agency continues to serve the historical V1 model, whose earlier travel accuracy was 66.5%. Do not describe the V2 result or V2 timing as active agency behavior.

## Public lanes, reported separately

| Lane | Engine | Agreement | Macro F1 | Failed decisions | Median ms |
|---|---|---:|---:|---:|---:|
| typed-independent | local | 34.05% | 0.2016 | 35/2000 | 1356.4 |
| typed-independent | jev | 73.65% | 0.6082 | 0/2000 | 158.7 |
| btzsc-independent | local | 49.67% | 0.4589 | 0/300 | 58.3 |
| btzsc-independent | jev | 76.33% | 0.7585 | 0/300 | 160.0 |

Independent normalized candidate-choice references, not the official native typed harness. Upstream exposure may apply. Failed decisions stay in denominators.

## Speed boundaries

Fresh travel median: local 98.6 ms; Jev 165.6 ms. Local p95 147.3 ms; Jev 230.4 ms.
Warm local tokenization and synchronized inference versus remote request wall time. Loading, HTTP app calls and browser rendering are separate.

The earlier development precision experiment used alternating independently reloaded FP32/FP16 blocks. It found zero changed choices over 2,176 development decisions and roughly halved warm model latency. That optimization was selected before final generation. See experiments/v2/runs/deeper-precision-benchmark/report.json.

## Reference provenance

Synthetic fact-first references with agent factual repairs, blind Gemma audits and two openly recorded writer-assisted adjudications.
All original drafts and audits, 52 factual/wording repairs, seven unchanged re-audits and two non-unanimous adjudications are preserved. Prespecified labels never changed; all 360 cases remained. Qwen was the writer and the two-case second-opinion model, so those judgments are not independent of the writer. No human validation.

## Reproduction

The source hashes are in frozen-v2-summary.json. The complete resource includes the frozen challenger even though it was not promoted. Final and public results did not tune weights, thresholds, input policy or the seed. Repeat scoring incurs Jev API calls; use preserved raw receipts for report reproduction.
