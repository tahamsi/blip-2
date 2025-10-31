from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import Blip2ForConditionalGeneration, Blip2Model, Blip2Processor, PreTrainedModel

DEFAULT_MODEL_ID = "Salesforce/blip2-opt-2.7b"

_device: torch.device
_dtype: torch.dtype


def _set_device(device: torch.device) -> None:
    global _device, _dtype
    _device = device
    _dtype = torch.float16 if device.type == "cuda" else torch.float32


_set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

if _device.type != "cuda":
    print("CUDA not available; running on CPU.", flush=True)


def get_runtime_device() -> torch.device:
    return _device


def get_runtime_dtype() -> torch.dtype:
    return _dtype


def set_runtime_device(device: torch.device) -> None:
    _set_device(device)


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_processor(model_source: str) -> Blip2Processor:
    source_path = Path(model_source).expanduser()
    if source_path.exists():
        return Blip2Processor.from_pretrained(str(source_path))

    try:
        return Blip2Processor.from_pretrained(model_source, local_files_only=True)
    except OSError:
        print(f"Local processor files for '{model_source}' not found; attempting download...", flush=True)
        return Blip2Processor.from_pretrained(model_source)


def _load_model(model_cls: type[PreTrainedModel], model_source: str, *, dtype: torch.dtype) -> PreTrainedModel:
    source_path = Path(model_source).expanduser()
    if source_path.exists():
        return model_cls.from_pretrained(str(source_path), dtype=dtype)

    try:
        return model_cls.from_pretrained(model_source, dtype=dtype, local_files_only=True)
    except OSError:
        print(f"Local model weights for '{model_source}' not found; attempting download...", flush=True)
        return model_cls.from_pretrained(model_source, dtype=dtype)


@lru_cache()
def get_processor(model_source: str = DEFAULT_MODEL_ID) -> Blip2Processor:
    return _load_processor(model_source)


def move_model_to_runtime(model: PreTrainedModel) -> PreTrainedModel:
    try:
        return model.to(device=_device, dtype=_dtype).eval()
    except RuntimeError as exc:
        if _device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory; falling back to CPU.", flush=True)
            clear_cuda_cache()
            _set_device(torch.device("cpu"))
            return model.to(device=_device, dtype=_dtype).eval()
        raise


@lru_cache()
def get_caption_model(model_source: str = DEFAULT_MODEL_ID) -> Blip2ForConditionalGeneration:
    model = _load_model(Blip2ForConditionalGeneration, model_source, dtype=_dtype)
    return move_model_to_runtime(model)


@lru_cache()
def get_retrieval_model(model_source: str = DEFAULT_MODEL_ID) -> Blip2Model:
    model = _load_model(Blip2Model, model_source, dtype=_dtype)
    return move_model_to_runtime(model)


def move_tensors(inputs: Mapping[str, object]) -> dict[str, object]:
    def _move_all(*, non_blocking: bool) -> dict[str, object]:
        moved: dict[str, object] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
                continue
            if value.is_floating_point():
                moved[key] = value.to(device=_device, dtype=_dtype, non_blocking=non_blocking)
            else:
                moved[key] = value.to(device=_device, non_blocking=non_blocking)
        return moved

    try:
        return _move_all(non_blocking=_device.type == "cuda")
    except RuntimeError as exc:
        if _device.type == "cuda" and "out of memory" in str(exc).lower():
            print("CUDA ran out of memory while preparing inputs; falling back to CPU.", flush=True)
            clear_cuda_cache()
            _set_device(torch.device("cpu"))
            return _move_all(non_blocking=False)
        raise


def prepare_inputs(
    image: Image.Image,
    processor: Blip2Processor,
    *,
    text: Sequence[str] | str | None = None,
) -> dict[str, object]:
    processor_kwargs: dict[str, object] = {"return_tensors": "pt"}

    if text is not None:
        processor_kwargs["text"] = text
        if isinstance(text, Sequence) and not isinstance(text, str):
            processor_kwargs["padding"] = True

    tensors = processor(images=image, **processor_kwargs)
    return dict(tensors.items())


def _load_manifest_rows(manifest_path: str, *, expected_fields: int) -> list[tuple[str, ...]]:
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[tuple[str, ...]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) != expected_fields:
                raise ValueError(
                    f"Expected {expected_fields} tab-delimited fields on line {line_number} of {path}; "
                    f"found {len(parts)}.",
                )
            rows.append(tuple(parts))

    if not rows:
        raise ValueError(f"No annotations found in manifest: {path}")
    return rows


def _mask_pad_tokens(labels: torch.Tensor, *, pad_token_id: int | None) -> torch.Tensor:
    if pad_token_id is None:
        return labels
    masked = labels.clone()
    masked[masked == pad_token_id] = -100
    return masked


class ImageCaptionDataset(Dataset):
    """Dataset wrapper for image-caption fine-tuning."""

    def __init__(
        self,
        annotations: Sequence[tuple[str, str]],
        processor: Blip2Processor,
        *,
        max_length: int,
    ) -> None:
        self._processor = processor
        self._max_length = max_length
        self._entries = [(Path(image_path).expanduser(), caption) for image_path, caption in annotations]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        image_path, caption = self._entries[idx]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            encoding = self._processor(
                images=image,
                text=str(caption),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )

        tensors = {key: value.squeeze(0) for key, value in encoding.items()}
        pad_token_id = self._processor.tokenizer.pad_token_id
        tensors["labels"] = _mask_pad_tokens(tensors["input_ids"], pad_token_id=pad_token_id)
        return tensors


class ImageVQADataset(Dataset):
    """Dataset wrapper for visual question answering fine-tuning."""

    def __init__(
        self,
        annotations: Sequence[tuple[str, str, str]],
        processor: Blip2Processor,
        *,
        max_length: int,
    ) -> None:
        self._processor = processor
        self._max_length = max_length
        self._entries = [
            (Path(image_path).expanduser(), question, answer)
            for image_path, question, answer in annotations
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        image_path, question, answer = self._entries[idx]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            encoding = self._processor(
                images=image,
                text=str(question),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )

        tensors = {key: value.squeeze(0) for key, value in encoding.items()}
        target = self._processor.tokenizer(
            str(answer),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
        )
        labels = target["input_ids"].squeeze(0)
        pad_token_id = self._processor.tokenizer.pad_token_id
        tensors["labels"] = _mask_pad_tokens(labels, pad_token_id=pad_token_id)
        return tensors


class ImageRetrievalDataset(Dataset):
    """Dataset wrapper for image-text retrieval fine-tuning."""

    def __init__(
        self,
        annotations: Sequence[tuple[str, str]],
        processor: Blip2Processor,
        *,
        max_length: int,
    ) -> None:
        self._processor = processor
        self._max_length = max_length
        self._entries = [(Path(image_path).expanduser(), text) for image_path, text in annotations]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        image_path, text = self._entries[idx]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            encoding = self._processor(
                images=image,
                text=str(text),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
            )

        return {key: value.squeeze(0) for key, value in encoding.items()}


def _build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_blip2(
    train_manifest: str,
    *,
    val_manifest: str | None,
    output_dir: str,
    model_source: str,
    task: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_length: int,
    num_workers: int,
) -> dict[str, float | None | str]:
    """Fine-tune BLIP-2 for captioning, VQA, or retrieval and persist the resulting checkpoint."""
    processor = get_processor(model_source)
    task_normalized = task.lower()

    if task_normalized == "caption":
        train_rows = _load_manifest_rows(train_manifest, expected_fields=2)
        train_dataset: Dataset = ImageCaptionDataset(train_rows, processor, max_length=max_length)
        model = _load_model(Blip2ForConditionalGeneration, model_source, dtype=_dtype)
        val_dataset: Dataset | None = None
        if val_manifest is not None:
            val_rows = _load_manifest_rows(val_manifest, expected_fields=2)
            val_dataset = ImageCaptionDataset(val_rows, processor, max_length=max_length)
    elif task_normalized == "vqa":
        train_rows_triplets = _load_manifest_rows(train_manifest, expected_fields=3)
        train_dataset = ImageVQADataset(train_rows_triplets, processor, max_length=max_length)
        model = _load_model(Blip2ForConditionalGeneration, model_source, dtype=_dtype)
        val_dataset = None
        if val_manifest is not None:
            val_rows_triplets = _load_manifest_rows(val_manifest, expected_fields=3)
            val_dataset = ImageVQADataset(val_rows_triplets, processor, max_length=max_length)
    elif task_normalized == "retrieval":
        train_rows_pairs = _load_manifest_rows(train_manifest, expected_fields=2)
        train_dataset = ImageRetrievalDataset(train_rows_pairs, processor, max_length=max_length)
        model = _load_model(Blip2Model, model_source, dtype=_dtype)
        val_dataset = None
        if val_manifest is not None:
            val_rows_pairs = _load_manifest_rows(val_manifest, expected_fields=2)
            val_dataset = ImageRetrievalDataset(val_rows_pairs, processor, max_length=max_length)
    else:
        raise ValueError(
            f"Unsupported task '{task}'. Expected one of: caption, vqa, retrieval.",
        )

    model = model.to(device=_device, dtype=_dtype)
    train_loader = _build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader: DataLoader | None = None
    if val_dataset is not None:
        val_loader = _build_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(enabled=_device.type == "cuda")

    final_train_loss = 0.0
    final_val_loss: float | None = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            inputs = move_tensors(batch)

            try:
                with autocast(device_type=_device.type, dtype=_dtype, enabled=_device.type == "cuda"):
                    if task_normalized in {"caption", "vqa"}:
                        outputs = model(**inputs)
                        loss = outputs.loss
                    else:
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        logits_per_text = outputs.logits_per_text
                        targets = torch.arange(
                            logits_per_image.size(0),
                            device=_device,
                            dtype=torch.long,
                        )
                        loss = (
                            F.cross_entropy(logits_per_image, targets)
                            + F.cross_entropy(logits_per_text, targets)
                        ) / 2.0
            except RuntimeError as exc:
                if _device.type == "cuda" and "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "CUDA ran out of memory during training. Consider reducing batch size or sequence length.",
                    ) from exc
                raise

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        final_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} - train loss: {final_train_loss:.4f}", flush=True)

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = move_tensors(batch)
                    with autocast(
                        device_type=_device.type,
                        dtype=_dtype,
                        enabled=_device.type == "cuda",
                    ):
                        if task_normalized in {"caption", "vqa"}:
                            outputs = model(**inputs)
                            loss = outputs.loss
                        else:
                            outputs = model(**inputs)
                            logits_per_image = outputs.logits_per_image
                            logits_per_text = outputs.logits_per_text
                            targets = torch.arange(
                                logits_per_image.size(0),
                                device=_device,
                                dtype=torch.long,
                            )
                            loss = (
                                F.cross_entropy(logits_per_image, targets)
                                + F.cross_entropy(logits_per_text, targets)
                            ) / 2.0
                    val_loss_total += loss.item()

            final_val_loss = val_loss_total / len(val_loader)
            print(f"Epoch {epoch}/{num_epochs} - val loss: {final_val_loss:.4f}", flush=True)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"Saved fine-tuned model and processor to {output_path}", flush=True)

    return {
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "output_dir": str(output_path),
    }
