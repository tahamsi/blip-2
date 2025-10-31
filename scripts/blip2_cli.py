from __future__ import annotations

import argparse

from models.blip2_runtime import DEFAULT_MODEL_ID, train_blip2
from scripts.blip2_interfaces import (
    blip2_caption,
    blip2_image_text_retrieval,
    blip2_vqa,
)


def _format_retrieval_results(results: list[tuple[str, float]]) -> str:
    lines = []
    for rank, (text, score) in enumerate(results, start=1):
        lines.append(f"{rank:>2}: {score:.3f} :: {text}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLIP-2 utilities: fine-tuning, captioning, VQA, and image-text retrieval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    caption_parser = subparsers.add_parser("caption", help="Generate a caption for an image.")
    caption_parser.add_argument("image_path", help="Path to the image file.")
    caption_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate for the caption (default: 50).",
    )
    caption_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help=(
            "Model identifier or local path to use for captioning "
            f"(default: {DEFAULT_MODEL_ID})."
        ),
    )

    vqa_parser = subparsers.add_parser("vqa", help="Answer a question about an image.")
    vqa_parser.add_argument("image_path", help="Path to the image file.")
    vqa_parser.add_argument("question", help="Question to ask about the image.")
    vqa_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum number of tokens to generate for the answer (default: 30).",
    )
    vqa_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help=(
            "Model identifier or local path to use for VQA "
            f"(default: {DEFAULT_MODEL_ID})."
        ),
    )

    retrieval_parser = subparsers.add_parser("retrieval", help="Rank candidate texts for an image.")
    retrieval_parser.add_argument("image_path", help="Path to the image file.")
    retrieval_parser.add_argument(
        "texts",
        nargs="+",
        help="Candidate texts to score against the image.",
    )
    retrieval_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help=(
            "Model identifier or local path to use for retrieval "
            f"(default: {DEFAULT_MODEL_ID})."
        ),
    )

    train_parser = subparsers.add_parser("train", help="Fine-tune BLIP-2 on a dataset.")
    train_parser.add_argument("train_manifest", help="Path to training manifest.")
    train_parser.add_argument(
        "--task",
        choices=["caption", "vqa", "retrieval"],
        default="caption",
        help="Task to fine-tune: caption, vqa, or retrieval (default: caption).",
    )
    train_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store the fine-tuned model and processor.",
    )
    train_parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_ID,
        help=(
            "Base model identifier or local checkpoint path to start fine-tuning from "
            f"(default: {DEFAULT_MODEL_ID})."
        ),
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs (default: 1).",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training and validation (default: 2).",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW (default: 1e-5).",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW (default: 0.0).",
    )
    train_parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum sequence length for padding/truncation (default: 50).",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    train_parser.add_argument(
        "--val-manifest",
        help="Optional validation manifest.",
    )

    args = parser.parse_args()

    if args.command == "caption":
        caption = blip2_caption(
            args.image_path,
            max_new_tokens=args.max_new_tokens,
            model_source=args.model_source,
        )
        print(f"Caption: {caption}")
    elif args.command == "vqa":
        answer = blip2_vqa(
            args.image_path,
            args.question,
            max_new_tokens=args.max_new_tokens,
            model_source=args.model_source,
        )
        print(f"Answer: {answer}")
    elif args.command == "retrieval":
        results = blip2_image_text_retrieval(
            args.image_path,
            args.texts,
            model_source=args.model_source,
        )
        print(_format_retrieval_results(results))
    elif args.command == "train":
        stats = train_blip2(
            args.train_manifest,
            val_manifest=args.val_manifest,
            output_dir=args.output_dir,
            model_source=args.model_source,
            task=args.task,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_length=args.max_length,
            num_workers=args.num_workers,
        )
        train_loss = stats["train_loss"]
        val_loss = stats["val_loss"]
        summary = f"Training complete. Final train loss: {train_loss:.4f}"
        if isinstance(val_loss, float):
            summary += f" | Final val loss: {val_loss:.4f}"
        print(summary, flush=True)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
