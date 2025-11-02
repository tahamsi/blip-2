# BLIP-2 Training & Inference Utilities

Pre-trained vision-language models (VLMs) pair visual encoders with large language models to understand and generate grounded text. They enable zero-shot and few-shot transfer to captioning, VQA, and retrieval tasks without training from scratch on massive multimodal corpora.

[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) (Salesforce Research, 2023) introduced a two-stage, parameter-efficient VLM that bridges a frozen vision backbone to either OPT, FlanT5, or Vicuna language models. The official implementation and checkpoints are available at the [Salesforce LAVIS repository](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

This project exposes a command-line interface around the BLIP-2 family of models for three core tasks:

- **Captioning** – generate free-form descriptions for images.
- **Visual Question Answering (VQA)** – answer natural-language questions about images.
- **Image-Text Retrieval** – rank candidate texts by relevance to a given image.

The CLI supports both inference with any Hugging Face checkpoint and fine-tuning on your own data. The project is organised as:

- `scripts/blip2_cli.py` – command-line entry point.
- `scripts/blip2_interfaces.py` – task-oriented inference helpers (captioning, VQA, retrieval).
- `models/blip2_runtime.py` – model loading, dataset utilities, and fine-tuning logic.
- `data/` – store training/validation manifests and sample assets here.
- `models/` – recommended location for saved checkpoints.

---

## Environment Setup

1. Create and activate a Python environment (Python 3.10+ recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. From the repository root, set `PYTHONPATH` so Python can resolve the local `scripts/` and `models/` packages:

   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```

---

## Data Preparation

Fine-tuning uses simple tab-separated-value (TSV) manifests. Each non-empty, non-comment line (lines starting with `#` are ignored) represents one training example. Store manifests under `data/` (for example `data/caption/train_manifest.tsv`), and make sure image paths are absolute or relative to where you run the CLI.

### Captioning

`image_path<TAB>caption_text`

Example:

```
/data/coco/train2017/000000000009.jpg	A man riding a bike down a city street.
/data/coco/train2017/000000000025.jpg	A child is flying a kite in a park.
```

### Visual Question Answering (VQA)

`image_path<TAB>question_text<TAB>answer_text`

Example:

```
/data/vqa/image1.jpg	What color is the bus?	Yellow
/data/vqa/image2.jpg	How many dogs are in the photo?	Three
```

### Image-Text Retrieval

`image_path<TAB>candidate_text`

Example:

```
/data/flickr/img1.jpg	A group of friends hiking through a forest trail.
/data/flickr/img1.jpg	A family eating dinner at a restaurant.
```

> Tip: You can generate manifests programmatically:
>
> ```python
> from pathlib import Path
>
> entries = [
>     ("/data/img1.jpg", "A scenic mountain view."),
>     ("/data/img2.jpg", "Sunset over the lake."),
> ]
>
> with Path("train_manifest.tsv").open("w", encoding="utf-8") as f:
>     for image_path, caption in entries:
>         f.write(f"{image_path}\t{caption}\n")
> ```

---

## Command-Line Usage

All commands share the same entry point:

```bash
python scripts/blip2_cli.py <command> [options]
```

Run `--help` on the script or any subcommand for full details.

### Quick Checks

```bash
python scripts/blip2_cli.py --help
python scripts/blip2_cli.py caption --help
python scripts/blip2_cli.py train --help
```

### Caption Generation

```bash
python scripts/blip2_cli.py caption data/samples/image.jpg \
    --max-new-tokens 60 \
    --model-source Salesforce/blip2-opt-2.7b
```

### VQA Inference

```bash
python scripts/blip2_cli.py vqa data/samples/image.jpg \
    "What is the person holding?" \
    --max-new-tokens 30 \
    --model-source Salesforce/blip2-opt-2.7b
```

### Retrieval Inference

```bash
python scripts/blip2_cli.py retrieval data/samples/image.jpg \
    "A chef cooking in a kitchen." \
    "A runner crossing the finish line." \
    --model-source Salesforce/blip2-opt-2.7b
```

Behind the scenes the image tensor is automatically replicated so each candidate text is scored independently. If CUDA kernels fault (for example due to masked scatter asserts) the CLI transparently reloads the model on CPU and retries. Candidate rankings are derived from the average log-likelihood of each decoded sequence, so you can pass any number of texts and receive a normalized probability distribution over them.

---

## Fine-Tuning

Fine-tuned checkpoints are saved alongside the processor so they can be reused later via `--model-source /path/to/output_dir`.

```bash
python scripts/blip2_cli.py train data/caption/train_manifest.tsv \
    --task caption \
    --output-dir models/blip2-caption-finetuned \
    --model-source Salesforce/blip2-opt-2.7b \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-6 \
    --weight-decay 0.01 \
    --max-length 50 \
    --num-workers 4 \
    --val-manifest data/caption/val_manifest.tsv
```

Change `--task` to `vqa` or `retrieval` when training those objectives (remember to supply the corresponding manifest format described above).

### Using the Fine-Tuned Weights

```bash
python scripts/blip2_cli.py caption data/samples/image.jpg \
    --model-source models/blip2-caption-finetuned
```

The same `--model-source` flag works for `vqa` and `retrieval`.

---

## Notes & Recommendations

- GPU acceleration is automatically used when available; the script gracefully falls back to CPU if CUDA runs out of memory.
- For retrieval training, the loss is symmetric cross entropy over image-to-text and text-to-image logits.
- Keep manifests small when sanity-checking new pipelines—try `--epochs 1 --batch-size 1` first.
- If sandboxed environments block bytecode creation, you can verify syntax with `PYTHONDONTWRITEBYTECODE=1 python -m py_compile scripts/blip2_cli.py`.

With these steps you have a full workflow: prepare data, fine-tune per task, and run inference against either the base BLIP-2 model or your custom checkpoint.
