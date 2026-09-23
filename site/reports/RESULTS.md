# Travel Lab: measured results

Run `overnight-20260920`. Selected checkpoint `travel-nli`.

## Comparison status

Fresh Jev API calls and local predictions use the same frozen state, question and candidate descriptions. Results apply to these inputs and deployment conditions, not universal model quality.

## Results

| Model / lane | Agreement | Macro F1 by field | p50 ms | Failed decisions |
|---|---:|---:|---:|---:|
| modernbert-nli-baseline-typed-independent | 0.417 | 0.241 | 355.2 | 0 |
| jev-travel-challenge | 0.968 | 0.962 | 171.3 | 0 |
| modernbert-nli-baseline-travel-challenge | 0.618 | 0.526 | 221.5 | 0 |
| travel-nli-travel-challenge | 0.815 | 0.774 | 110.9 | 0 |
| jev-typed-independent | 0.738 | 0.610 | 177.2 | 0 |
| frozen-head-travel-test | 0.376 | 0.228 | 120.2 | 0 |
| travel-nli-btzsc-independent | 0.797 | 0.771 | 30.2 | 0 |
| base-trained-typed-independent | 0.348 | 0.224 | 360.3 | 25 |
| modernbert-nli-baseline-travel-test | 0.501 | 0.467 | 135.5 | 0 |
| base-trained-travel-test | 0.350 | 0.199 | 122.9 | 0 |
| base-trained-travel-challenge | 0.292 | 0.180 | 131.6 | 0 |
| jev-btzsc-independent | 0.760 | 0.753 | 177.7 | 0 |
| modernbert-nli-baseline-btzsc-independent | 0.797 | 0.770 | 28.3 | 0 |
| travel-nli-travel-test | 0.665 | 0.605 | 102.9 | 0 |
| jev-travel-test | 0.963 | 0.958 | 173.7 | 0 |
| travel-nli-typed-independent | 0.422 | 0.245 | 347.7 | 0 |
| base-trained-btzsc-independent | 0.187 | 0.144 | 37.2 | 0 |

## Boundaries

- Travel references are synthetic, generated from a small template grammar, not human-reviewed real inventory.
- Each split uses one prose family per class/task; lexical family coverage is narrow. Scenario bootstrap intervals are conditional on this corpus and do not measure template-population uncertainty.
- All scenarios recombine four three-way policy dimensions: at most 81 semantic combinations per split. Counts do not imply 1200 independent real-world situations.
- Independent typed decisions measures agreement with a roughly 4B teacher, not verified truth.
- NLI starting weights were previously trained on a classification mixture including AG News, Banking77 and emotion domains. Public pilot results are not claims of untouched zero-shot task transfer.
- Jev results, where present, are fresh observational API calls against frozen identical candidate-choice inputs. No Jev outputs inform local training or changes. This standardized adapter is not the official native typed harness.
- Local latency includes tokenization and synchronized device inference, excludes loading and visual reveal. Local electricity and hardware are not free.
- The final model is frozen before final test scoring; final errors are not used to tune it.

## Reproduction

Use the pinned uv.lock, dataset manifests, model hashes and raw predictions. Every inference receives only state, instructions and candidate descriptions. Gold, oracle and factors are scoring-only fields. Calibration uses the dedicated calibration split.

## Costs

This run used existing local hardware and the existing Jev API account. No GPU rental. Jev cost estimates use reported input tokens at the verified $0.042 per million input tokens; they are not billing invoices. Electricity and machine depreciation were not instrumented, so local inference is not described as free.

## Human audit

No human label review has occurred. A separate 50-case audit packet is provided; reviewer cells are blank.
