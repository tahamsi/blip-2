from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image

from models.blip2_runtime import (
    DEFAULT_MODEL_ID,
    clear_cuda_cache,
    get_caption_model,
    get_processor,
    get_retrieval_model,
    get_runtime_device,
    move_tensors,
    prepare_inputs,
    reset_runtime_device,
)


def _encode_inputs(
    path: Path,
    processor,
    *,
    text: Sequence[str] | str | None = None,
):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return prepare_inputs(image, processor, text=text)


def blip2_caption(
    image_path: str,
    *,
    max_new_tokens: int = 50,
    model_source: str = DEFAULT_MODEL_ID,
) -> str:
    """Generate a caption for an image using BLIP-2."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    processor = get_processor(model_source)
    raw_inputs = _encode_inputs(path, processor)
    model = get_caption_model(model_source)
    inputs = move_tensors(raw_inputs)

    device = get_runtime_device()
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError:
        if device.type == "cuda":
            print("Encountered a CUDA error during captioning; retrying on CPU.", flush=True)
            reset_runtime_device(torch.device("cpu"))
            processor = get_processor(model_source)
            raw_inputs = _encode_inputs(path, processor)
            inputs = move_tensors(raw_inputs)
            model = get_caption_model(model_source)
            clear_cuda_cache()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            raise

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def blip2_vqa(
    image_path: str,
    question: str,
    *,
    max_new_tokens: int = 30,
    model_source: str = DEFAULT_MODEL_ID,
) -> str:
    """Answer a visual question for an image."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if not question.strip():
        raise ValueError("Question must be a non-empty string.")

    processor = get_processor(model_source)
    raw_inputs = _encode_inputs(path, processor, text=question)
    model = get_caption_model(model_source)
    inputs = move_tensors(raw_inputs)

    device = get_runtime_device()
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError:
        if device.type == "cuda":
            print("Encountered a CUDA error during VQA; retrying on CPU.", flush=True)
            reset_runtime_device(torch.device("cpu"))
            processor = get_processor(model_source)
            raw_inputs = _encode_inputs(path, processor, text=question)
            inputs = move_tensors(raw_inputs)
            model = get_caption_model(model_source)
            clear_cuda_cache()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            raise

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def blip2_image_text_retrieval(
    image_path: str,
    texts: Iterable[str],
    *,
    model_source: str = DEFAULT_MODEL_ID,
) -> list[tuple[str, float]]:
    """Rank candidate texts by relevance to the image."""
    text_list = [candidate.strip() for candidate in texts if candidate.strip()]
    if not text_list:
        raise ValueError("At least one non-empty candidate text is required.")

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    processor = get_processor(model_source)
    raw_inputs = _encode_inputs(path, processor, text=text_list)
    model = get_retrieval_model(model_source)
    inputs = move_tensors(raw_inputs)

    device = get_runtime_device()
    try:
        with torch.inference_mode():
            outputs = model(**inputs)
    except RuntimeError:
        if device.type == "cuda":
            print("Encountered a CUDA error during retrieval; retrying on CPU.", flush=True)
            reset_runtime_device(torch.device("cpu"))
            processor = get_processor(model_source)
            raw_inputs = _encode_inputs(path, processor, text=text_list)
            inputs = move_tensors(raw_inputs)
            model = get_retrieval_model(model_source)
            clear_cuda_cache()
            with torch.inference_mode():
                outputs = model(**inputs)
        else:
            raise

    logits = outputs.logits
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    if attention_mask is None:
        shift_mask = torch.ones_like(shift_labels, dtype=logits.dtype, device=logits.device)
    else:
        shift_mask = attention_mask[:, 1:].to(logits.device)

    log_probs = shift_logits.log_softmax(dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_mask = shift_mask.to(logits.dtype)
    token_log_probs = gathered * token_mask

    token_counts = token_mask.sum(dim=1).clamp_min(1e-6)
    average_log_probs = token_log_probs.sum(dim=1) / token_counts
    probabilities = torch.softmax(average_log_probs, dim=-1)

    ranked_indices = torch.argsort(probabilities, descending=True)
    return [(text_list[i], probabilities[i].item()) for i in ranked_indices]
