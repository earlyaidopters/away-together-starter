# Build your first specialist, from download to a tested result

This is the follow-along path for the video. Start from the repository root, opened as your project folder in Codex. The supplied example is travel. You can follow the concepts without ML experience; installing tools and checking your labels still require attention. Codex can help write and run the code, but it cannot certify that your business answers are correct.

## 1. Choose a classifier that already does something useful

Open [the author's model card](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0/tree/d421c4545a438fd006fb43f8b981c5d908faa1e1). Our starting point is **MoritzLaurer/ModernBERT-base-zeroshot-v2.0**, a ModernBERT model already adapted by its author for classification. This differs from the raw `answerdotai/ModernBERT-base` language encoder and from our later DeBERTa challenger. Check the task, weights, usage example, terms and measured results. Popularity alone does not establish suitability.

In our saved travel development run, this classifier achieved **74.75% accuracy before our travel training**. The selected travel checkpoint reached **93.63% on the same 800 development decisions**. These are saved development measurements used in model selection, not unseen final performance. The active checkpoint's historical final-test result was 66.5% on its separate older test. That gap is why you need all the later testing steps. Sources: `apps/agency/models/selection.json` and `runs/nli-baseline-dev.json` in the restored artifact. These scores do not measure images or Jev performance.

## 2. Get the source and install its tools

The repository stays private until video go-live. Follow [SETUP](SETUP.md) to get access and install Python 3.12, uv, Node 22+ and the matching source/model bundle. Public access must be verified at launch. Clone the one repository, then install the locked project environment. While private, GitHub must be authenticated with access; at go-live, use the same repository after public access is verified.

```bash
gh repo clone earlyaidopters/away-together
cd away-together
cd apps/agency
uv sync --frozen
cd ../..
```

Expected result: a project environment in `apps/agency/.venv`. The commands below use that environment through `uv run --project apps/agency`. The tutorial was checked on Apple silicon; other accelerator paths are implemented but not validated by that check. Read the hardware limitations in SETUP before downloading the separate image model.

The repository root means the folder containing README.md, apps and tools. Open that folder in Codex. A terminal is the box where you paste commands; press Enter to run one and wait for it to finish before the next. If a command fails, give Codex the command and complete error before continuing.

If the setup is unfamiliar, give Codex this first:

```text
Help me set up this repository on my computer. Check whether GitHub CLI, Python 3.12, uv and Node 22 or later are installed. Explain anything missing, use official installation instructions for my operating system, and verify each tool before running the commands in docs/STEP-BY-STEP.md. Open the repository root and tell me where to paste the first command. Do not start training or paid API calls yet.
```

## 3. Download the exact starting weights

```bash
uv run --project apps/agency python tools/tutorial.py download
```

Expected result: a printed model ID, revision and local cache path. The command downloads the tokenizer, configuration and safe weight files for the pinned revision. A repeat run may reuse the cache. This step does not train anything. First download needs internet; inference can use the cached files afterward. Preserve model attribution and applicable component terms.

## 4. Give Codex the complete job

Open [TRAIN-MY-SPECIALIST](../prompts/TRAIN-MY-SPECIALIST.md). Replace the bracketed job, input, questions and answers. Paste the entire prompt into Codex with this repository open. The website highlights those editable parts and the testing requirements. It is a prepared example specification, not a historical one-shot build transcript.

For travel, define each question separately. A cash refund is different from hotel credit. Missing evidence needs its own label. The model reads terms; arithmetic code checks budgets; a separate pretrained model reads photos.

## 5. Inspect a complete example and keep an untouched test

Open `apps/agency/data/train.jsonl`. A record contains a scenario `id`, `state` (the text), and `questions`. Each question contains its `id`, question, three candidate sentences, labels and a `gold` index. Index 0 refers to the first candidate. Keep descriptions and indexes aligned.

For another domain, the website's support editor exports checked `id/text/label` examples. `tools/prepare_task.py` converts those into the candidate-scoring schema. Create separate `train.jsonl`, `dev.jsonl` and `test.jsonl` files in your own data folder. Keep related conversations and templates together. There is no universal number of training examples that guarantees quality. Our synthetic travel data demonstrates the process; it is not customer research or human validation.

## 6. Make a separate workspace and prove the pipeline runs

```bash
python3 tools/tutorial.py prepare --run workshops/travel-smoke
uv run --project apps/agency python tools/tutorial.py baseline --run workshops/travel-smoke --limit 8
uv run --project apps/agency python tools/tutorial.py train --run workshops/travel-smoke --limit 8 --epochs 1
uv run --project apps/agency python tools/tutorial.py test --run workshops/travel-smoke --limit 8
```

The first command validates schema and exact split overlap, copies data, records hashes and creates an exclusive output directory. It refuses an existing directory. The next three run real model inference, a tiny real training pass and a paired evaluation. Eight decisions only check that the pipeline works. Do not use those scores as evidence of quality. For your data, add `--data path/to/your-data` to `prepare`. Near-duplicate review still requires judgment.

## 7. Measure the baseline, then train the full dataset

Create a new workspace so smoke outputs stay separate:

```bash
python3 tools/tutorial.py prepare --run workshops/my-travel-model
uv run --project apps/agency python tools/tutorial.py baseline --run workshops/my-travel-model
uv run --project apps/agency python tools/tutorial.py train --run workshops/my-travel-model --epochs 2
```

Expected baseline file: `workshops/my-travel-model/baseline.json`. Expected training outputs: `checkpoint/model.safetensors`, `checkpoint/tokenizer/`, `checkpoint/base-config/`, `training.json` and `freeze.json`.

The tutorial updates the last two encoder layers and the classification head. A batch is eight decisions; an epoch is one pass through the practice data; learning rate controls adjustment size. It selects the checkpoint with the best development macro-F1. This is a new tutorial recipe, not an exact re-execution of the original gradient-accumulation experiment. Results can differ. It does not change the live demo, charge Jev, or train on your final test.

Read each epoch's loss and development scores. Loss is a training error signal; a smaller loss alone does not prove better real-world answers. If memory fails, keep the failed folder and ask Codex to adjust a documented new run. Overlong input is rejected instead of silently chopped off.

## 8. Open the final exam once

```bash
uv run --project apps/agency python tools/tutorial.py test --run workshops/my-travel-model
```

Expected file: `test.json`, containing original and trained predictions for the same held-out decisions. Inspect all mistakes. Accuracy is the share matching the reference labels. Macro-F1 gives each label class equal weight before averaging; a majority class cannot dominate it as easily. If the specialist loses, keep the loss. New training changes need a new final test before another independent claim. The tool refuses to replace an existing final report and verifies checkpoint/data hashes.

## 9. Try a new piece of text

Save this as `my-policy.json` in the repository root:

```json
{
  "state": "Cancel at least seven days before arrival for a full refund to your original payment card.",
  "candidates": [
    "A full cash refund is available when cancelling before the deadline.",
    "A full cash refund is not available even when cancelling before the deadline.",
    "It is not specified whether a full cash refund is available."
  ]
}
```

```bash
uv run --project apps/agency python tools/tutorial.py predict --run workshops/my-travel-model --input my-policy.json
```

Inspect the returned index and candidate sentence. Probabilities are uncalibrated. Even this clear example is a model prediction, not a guaranteed outcome. Change the policy to hotel credit and try again. This runs your newly trained checkpoint without touching the app's selected historical model.

## 10. Connect the answers to your application

Use [API-EXAMPLE](API-EXAMPLE.md) to inspect the supplied travel app. It has fixed travel questions, thresholds and rule mappings. Your new checkpoint needs an explicit adapter and fresh validation before using that serving path. Ask Codex to load the tutorial checkpoint in a separate service, expose only the required question contract, and test match/decline/review behavior. Use code for arithmetic and keep reasons inspectable.

## 11. Add actual image understanding

Follow SETUP's separate OpenJev launcher, then open the photo lab. Inspect the six fixed observation questions in `travel_lab/vision.py`, their answers, and how `apply_visual_requirements` maps those answers to travellers. The photo lab uses three visible-feature questions for its upload exercise. This is a separately pretrained vision model, not fine-tuning the text model to see. A new visual trait requires a new question, rule mapping and tests. Verify the missing-image path. A pond, visible stairs or a pool does not establish contract terms, safety or a whole accessible route.

## 12. Package your work and repeat for your own job

Keep the task contract, base revision, label rules, split hashes, training settings, checkpoint, test predictions and README together. Preserve failure cases and upstream credits. Keep private data and caches out of Git. Your reusable artifacts are the code, model settings and test process; each new domain needs checked examples and its own validation. Codex's build-time usage, hardware and electricity are distinct from local inference API charges.

If you stop here, you have completed download → baseline → real training → paired test → one new prediction. The app and image integration remain inspectable in the supplied example, with the exact adaptation boundaries stated above.
