"""ConvNeXt-Tiny full-parameter training benchmark for Intel XPU.

Image decoding and augmentation intentionally remain on CPU. Each step consumes a
host batch and includes transfer, forward, backward, and AdamW. Synthetic mode is
deterministic and does not download data or weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from ..harness import register_workload


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


@dataclass(frozen=True)
class ConvNeXtConfig:
    batch_size: int = 32
    image_size: int = 224
    num_classes: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    data_root: str = "data"
    workers: int = 4
    synthetic: bool = False
    pretrained: bool = True
    seed: int = 2026
    checkpoint: str = ""
    resume: str = ""

    @classmethod
    def from_options(cls, options: dict[str, str]) -> "ConvNeXtConfig":
        converters: dict[str, Callable[[str], Any]] = {
            "batch_size": int,
            "image_size": int,
            "num_classes": int,
            "learning_rate": float,
            "weight_decay": float,
            "data_root": str,
            "workers": int,
            "synthetic": _bool,
            "pretrained": _bool,
            "seed": int,
            "checkpoint": str,
            "resume": str,
        }
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown ConvNeXt options: {sorted(unknown)}")
        values = {key: converters[key](value) for key, value in options.items()}
        config = cls(**values)
        if config.batch_size < 1 or config.image_size < 1 or config.num_classes < 2:
            raise ValueError("batch_size/image_size must be positive and num_classes >= 2")
        return config


class _SyntheticBatches:
    def __init__(self, config: ConvNeXtConfig):
        generator = torch.Generator().manual_seed(config.seed)
        self.images = torch.randn(
            config.batch_size,
            3,
            config.image_size,
            config.image_size,
            generator=generator,
        ).contiguous(memory_format=torch.channels_last)
        self.targets = torch.randint(
            config.num_classes, (config.batch_size,), generator=generator
        )

    def __iter__(self) -> "_SyntheticBatches":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images, self.targets


def _real_batches(config: ConvNeXtConfig) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    dataset = datasets.CIFAR100(
        config.data_root, train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )
    while True:
        for images, targets in loader:
            yield images.contiguous(memory_format=torch.channels_last), targets


def _build_model(config: ConvNeXtConfig) -> torch.nn.Module:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = (
        ConvNeXt_Tiny_Weights.DEFAULT
        if config.pretrained and not config.synthetic
        else None
    )
    model = convnext_tiny(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(input_features, config.num_classes)
    return model


def _adamw_options(device: str) -> dict[str, bool]:
    is_xpu = device.startswith("xpu")
    return {"foreach": False, "capturable": is_xpu, "fused": is_xpu}


class _HTLayerNorm(torch.nn.LayerNorm):
    """Expose the contiguous layout returned by native LayerNorm backward."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            input.contiguous(),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )


class _HTLayerNorm2d(torch.nn.LayerNorm):
    """Expose the contiguous input layout required by native LayerNorm.

    Native CPU and XPU LayerNorm already make this conversion internally. Doing
    it explicitly keeps AOTAutograd's fake backward layout consistent with the
    contiguous grad_input returned by the native kernels.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.permute(0, 2, 3, 1).contiguous()
        output = torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return output.permute(0, 3, 1, 2)


def _prepare_native_layer_norm(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        replacement_type: type[torch.nn.LayerNorm] | None = None
        if isinstance(child, torch.nn.LayerNorm):
            if type(child).__name__ == "LayerNorm2d":
                replacement_type = _HTLayerNorm2d
            elif type(child) is torch.nn.LayerNorm:
                replacement_type = _HTLayerNorm
        if replacement_type is not None:
            replacement = replacement_type(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=child.bias is not None,
            )
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            module._modules[name] = replacement
            replaced += 1
        else:
            replaced += _prepare_native_layer_norm(child)
    return replaced


class ConvNeXtWorkload:
    name = "convnext"

    def __init__(
        self,
        variant: str,
        device: str,
        config: ConvNeXtConfig,
        *,
        model_builder: Callable[[ConvNeXtConfig], torch.nn.Module] = _build_model,
        batches: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        self.variant, self.device, self.config = variant, device, config
        torch.manual_seed(config.seed)
        self.model = model_builder(config)
        if variant == "ht":
            _prepare_native_layer_norm(self.model)
        self.model = self.model.to(memory_format=torch.channels_last).to(device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **_adamw_options(device),
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)
        source = batches or (
            _SyntheticBatches(config) if config.synthetic else _real_batches(config)
        )
        self.batches = iter(source)
        self.completed_steps = 0
        if config.resume:
            self.load_checkpoint(config.resume)

        train_step = self._train_step
        if variant in {"xpu-compiled", "ht"}:
            compile_kwargs: dict[str, Any] = {"backend": "inductor", "dynamic": False}
            if variant == "ht":
                compile_kwargs["options"] = {"xpu.use_native_layer_norm": True}
            train_step = torch.compile(train_step, **compile_kwargs)
        self._execute = train_step

    @property
    def units_per_step(self) -> float:
        return float(self.config.batch_size)

    def _train_step(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = self.device.startswith("xpu")
        with torch.autocast(
            device_type=self.device.split(":", 1)[0],
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            loss = self.loss_fn(self.model(images), targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def step(self) -> torch.Tensor:
        images, targets = next(self.batches)
        images = images.to(
            self.device,
            non_blocking=self.device.startswith("xpu"),
            memory_format=torch.channels_last,
        )
        targets = targets.to(self.device, non_blocking=self.device.startswith("xpu"))
        loss = self._execute(images, targets)
        self.completed_steps += 1
        return loss

    def residency_objects(self) -> list[Any]:
        return [self.model, self.optimizer, self.loss_fn]

    def benchmark_metadata(self) -> dict[str, Any]:
        metadata = asdict(self.config)
        metadata.pop("checkpoint")
        metadata.pop("resume")
        return metadata

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "completed_steps": self.completed_steps,
                "config": {
                    field.name: getattr(self.config, field.name)
                    for field in fields(self.config)
                },
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.completed_steps = int(state.get("completed_steps", 0))

    def finalize(self) -> None:
        if self.config.checkpoint:
            self.save_checkpoint(self.config.checkpoint)


def create_convnext(
    variant: str, device: str, options: dict[str, str]
) -> ConvNeXtWorkload:
    return ConvNeXtWorkload(variant, device, ConvNeXtConfig.from_options(options))


register_workload("convnext", create_convnext)
