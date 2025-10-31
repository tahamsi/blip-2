from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from models.blip2_runtime import (
    DEFAULT_MODEL_ID,
    clear_cuda_cache,
    get_caption_model,
    get_processor,
    get_retrieval_model,
    get_runtime_device,
    move_model_to_runtime,
    move_tensors,
    prepare_inputs,
    set_runtime_device,
)


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
    with Image.open(path) as image:
        image = image.convert("RGB")
        raw_inputs = prepare_inputs(image, processor)

    model = get_caption_model(model_source)
    inputs = move_tensors(raw_inputs)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        device = get_runtime_device()
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory during captioning; retrying on CPU.", flush=True)
            clear_cuda_cache()
            set_runtime_device(torch.device("cpu"))
            model = move_model_to_runtime(model)
            inputs = move_tensors(raw_inputs)
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
    with Image.open(path) as image:
        image = image.convert("RGB")
        raw_inputs = prepare_inputs(image, processor, text=question)

    model = get_caption_model(model_source)
    inputs = move_tensors(raw_inputs)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except RuntimeError as exc:
        device = get_runtime_device()
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory during VQA; retrying on CPU.", flush=True)
            clear_cuda_cache()
            set_runtime_device(torch.device("cpu"))
            model = move_model_to_runtime(model)
            inputs = move_tensors(raw_inputs)
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
    with Image.open(path) as image:
        image = image.convert("RGB")
        raw_inputs = prepare_inputs(image, processor, text=text_list)

    model = get_retrieval_model(model_source)
    inputs = move_tensors(raw_inputs)

    try:
        with torch.inference_mode():
            outputs = model(**inputs)
    except RuntimeError as exc:
        device = get_runtime_device()
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory during retrieval; retrying on CPU.", flush=True)
            clear_cuda_cache()
            set_runtime_device(torch.device("cpu"))
            model = move_model_to_runtime(model)
            inputs = move_tensors(raw_inputs)
            with torch.inference_mode():
                outputs = model(**inputs)
        else:
            raise

    logits = outputs.logits_per_image.squeeze(0)  # similarity scores
    probabilities = logits.softmax(dim=-1)

    ranked_indices = torch.argsort(probabilities, descending=True)
    return [(text_list[i], probabilities[i].item()) for i in ranked_indices]
