# Build Your Own Jev: the starter

**Learn the workflow. Try the demo. Train a small specialist.**

[Visual guide](https://build-your-own-jev.markkashef.chatgpt.site) · [Try the sample](https://build-your-own-jev.markkashef.chatgpt.site/demo/) · [Complete prompt](prompts/TRAIN-MY-SPECIALIST.md)

One model reads a holiday's terms. A separate image model reads its photos. Ordinary code compares their answers with each traveller's wish list. This free starter shows how that idea works and gives you a small training recipe to adapt.

## What's included?

- The interactive travel demo with 40 fictional offers and nine recorded photo observations.
- The complete reusable prompt and a plain-English guide.
- A pinned ModernBERT training recipe: prepare, baseline, train, test, predict.
- A small synthetic sample: 24 training, 12 development and 12 test scenarios.

The demo uses **recorded model outputs**; your browser recomputes budgets, preferences and photo-selection rules. It does not analyze new text or images. The starter recipe trains a separate educational model; it does not reproduce the video's V2 score.

## Run the sample app

```bash
git clone https://github.com/earlyaidopters/away-together-starter.git
cd away-together-starter/apps/public-demo
npm ci
npm run dev
```

Open the local address shown by Vite. No model download is needed for this path.

## Try a small training experiment

Start with Python 3.12 and uv. From the repository root:

```bash
cd apps/agency
uv sync --frozen
cd ../..
uv run --project apps/agency python tools/tutorial.py prepare --run workshops/first-check
uv run --project apps/agency python tools/tutorial.py download
uv run --project apps/agency python tools/tutorial.py baseline --run workshops/first-check --limit 8
uv run --project apps/agency python tools/tutorial.py train --run workshops/first-check --limit 8 --epochs 1
uv run --project apps/agency python tools/tutorial.py test --run workshops/first-check --limit 8
```

This downloads the pinned base model and writes only to the new workshop. It is a pipeline smoke test, **not an accuracy benchmark**. CPU, CUDA or Apple MPS is selected by the tutorial. Runtime and memory depend on your machine. No hosted model API is required; downloads and hardware have costs.

For your own task, replace the bracketed fields in [the prompt](prompts/TRAIN-MY-SPECIALIST.md). Prepare realistic, checked examples; separate related documents by split; measure the unmodified model; then train and freeze before the final test. Use a fresh workshop for each experiment. [Step-by-step commands](docs/STEP-BY-STEP.md).

## Starter versus community edition

| Free starter | Community edition |
|---|---|
| Public demo and visual guide | Complete working local agency |
| Full prompt and small training recipe | Full training pipeline and experiments |
| Small synthetic example dataset | Saved checkpoints and evaluation receipts |
| Learn and adapt independently | Exclusive training and community support |

The original experiment scored 95.28% versus Jev's 98.61% on the same synthetic travel test. It improved on the earlier model and **did not beat Jev**. See the guide for scope and limitations. Photos cannot prove refund rights, included access, or a fully accessible route.

Created by Mark Kashef. Original code is [MIT](LICENSE); [upstream terms](THIRD-PARTY-NOTICES.md) remain separate. Earlier public MIT releases retain their existing permissions. The community edition is the home for the complete package and future development.


## Build your own version with us

Get the complete project code, training pipeline, model checkpoints, and exclusive training inside Early AI Adopters.

[Join Early AI Adopters](https://www.skool.com/earlyaidopters/about)
