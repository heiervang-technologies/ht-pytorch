"""Qwen 3.5 supervised fine-tuning workload for Intel XPU.

Heavy third-party packages are imported only by the production factory methods so
the data, loss masking, checkpoint, and training-loop behavior remains unit
testable without downloading a model or dataset.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import torch
from torch import Tensor, nn

from .harness import register_workload


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DATASET_ID = "databricks/databricks-dolly-15k"
TRAIN_EXAMPLES = 4096
EVAL_EXAMPLES = 512
IGNORE_INDEX = -100


def _bool_option(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


@dataclass(frozen=True)
class QwenSFTConfig:
    """Configuration fixed by the benchmark-of-record.

    Tests and developer smoke runs may override the sizes, but reported benchmark
    results must use the defaults returned for their training mode.
    """

    training: Literal["lora", "full"] = "lora"
    model_id: str = MODEL_ID
    device: str = "xpu"
    sequence_length: int | None = None
    seed: int = 2026
    learning_rate: float | None = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int | None = None
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_backend: str = "inductor"
    optimization_variant: Literal["upstream", "ht"] = "upstream"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.training not in ("lora", "full"):
            raise ValueError("training must be 'lora' or 'full'")
        if self.optimization_variant not in ("upstream", "ht"):
            raise ValueError("optimization_variant must be 'upstream' or 'ht'")
        if self.sequence_length is None:
            object.__setattr__(
                self, "sequence_length", 512 if self.training == "lora" else 256
            )
        if self.gradient_accumulation_steps is None:
            # Microbatch is one; keep the effective token count at 2,048.
            object.__setattr__(
                self,
                "gradient_accumulation_steps",
                max(1, 2048 // self.sequence_length),
            )
        if self.learning_rate is None:
            object.__setattr__(
                self, "learning_rate", 2e-4 if self.training == "lora" else 2e-5
            )
        if self.sequence_length <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("sequence and accumulation lengths must be positive")

    @classmethod
    def from_options(
        cls, options: dict[str, str], *, device: str, compile_model: bool
    ) -> "QwenSFTConfig":
        converters: dict[str, Any] = {
            "training": str,
            "model_id": str,
            "sequence_length": int,
            "seed": int,
            "learning_rate": float,
            "weight_decay": float,
            "gradient_accumulation_steps": int,
            "gradient_checkpointing": _bool_option,
            "compile_backend": str,
            "lora_rank": int,
            "lora_alpha": int,
            "lora_dropout": float,
        }
        unknown = set(options) - {field.name for field in fields(cls)}
        # Device and compilation mode are controlled by the shared runner so a
        # result cannot accidentally be labelled as a different variant/device.
        unknown |= set(options) & {"device", "compile_model", "optimization_variant"}
        if unknown:
            raise ValueError(
                f"unknown or runner-controlled Qwen options: {sorted(unknown)}"
            )
        values = {key: converters[key](value) for key, value in options.items()}
        return cls(device=device, compile_model=compile_model, **values)


def format_dolly_prompt(example: Mapping[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    context = str(example.get("context", "")).strip()
    prompt = f"### Instruction:\n{instruction}\n\n"
    if context:
        prompt += f"### Context:\n{context}\n\n"
    return prompt + "### Response:\n"


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return list(ids)


def encode_response_only(
    example: Mapping[str, Any], tokenizer: Any, sequence_length: int
) -> dict[str, Tensor]:
    """Tokenize one Dolly row and mask prompt/padding tokens from the SFT loss."""

    prompt_ids = _encode(
        tokenizer, format_dolly_prompt(example), add_special_tokens=True
    )
    response = str(example.get("response", "")).strip()
    eos = tokenizer.eos_token or ""
    response_ids = _encode(tokenizer, response + eos, add_special_tokens=False)
    if not response_ids:
        raise ValueError("Dolly response produced no tokens")

    # Preserve response supervision when long instructions need truncation.
    response_ids = response_ids[:sequence_length]
    prompt_budget = max(0, sequence_length - len(response_ids))
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = (prompt_ids + response_ids)[:sequence_length]
    response_start = len(prompt_ids)
    attention_mask = [1] * len(input_ids)
    labels = [IGNORE_INDEX] * response_start + input_ids[response_start:]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define a pad or EOS token id")
    padding = sequence_length - len(input_ids)
    input_ids += [pad_id] * padding
    attention_mask += [0] * padding
    labels += [IGNORE_INDEX] * padding
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def deterministic_dolly_split(
    dataset: Any,
    *,
    seed: int = 2026,
    train_examples: int = TRAIN_EXAMPLES,
    eval_examples: int = EVAL_EXAMPLES,
) -> tuple[Any, Any]:
    required = train_examples + eval_examples
    if len(dataset) < required:
        raise ValueError(
            f"Dolly dataset has {len(dataset)} rows; {required} are required"
        )
    shuffled = dataset.shuffle(seed=seed)
    return (
        shuffled.select(range(train_examples)),
        shuffled.select(range(train_examples, required)),
    )


def load_dolly_splits(config: QwenSFTConfig) -> tuple[Any, Any]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train")
    return deterministic_dolly_split(dataset, seed=config.seed)


def load_tokenizer(config: QwenSFTConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: QwenSFTConfig) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if config.training == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
                modules_to_save=None,
            ),
        )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model.to(config.device)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class QwenSFTWorkload:
    """Small adapter consumed by the shared ``benchmarks.ht_xpu`` runner."""

    def __init__(
        self,
        config: QwenSFTConfig,
        *,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.tokenizer = tokenizer
        self.model = model if model is not None else load_model(config)
        self.model.to(config.device)
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("workload model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
            capturable=config.device.startswith("xpu"),
        )
        self._micro_step = 0
        self.train_dataset: Any | None = None
        self.eval_dataset: Any | None = None
        self.optimizer.zero_grad(set_to_none=True)
        self._batch_iterator: Iterator[dict[str, Tensor]] | None = None
        self._last_device_batch: dict[str, Tensor] = {}
        if config.compile_model:
            backend = config.compile_backend
            if config.optimization_variant == "ht":
                from benchmarks.ht_xpu.qwen_ops import (
                    enable_qwen35_compilation_paths,
                )

                enable_qwen35_compilation_paths(self.model)
                backend = "ht_xpu"
            self.model = torch.compile(
                self.model,
                backend=backend,
                dynamic=False,
                fullgraph=False,
            )

    @property
    def name(self) -> str:
        return f"qwen35-{self.config.training}"

    @property
    def units_per_step(self) -> float:
        return float(self.config.sequence_length * self.config.gradient_accumulation_steps)

    def step(self) -> dict[str, Tensor | bool]:
        """Execute one complete optimizer step for stable benchmark sampling."""

        result: dict[str, Tensor | bool] = {}
        for _ in range(self.config.gradient_accumulation_steps):
            if self._batch_iterator is None:
                self._batch_iterator = self.batches()
            try:
                batch = next(self._batch_iterator)
            except StopIteration:
                self._batch_iterator = self.batches()
                batch = next(self._batch_iterator)
            result = self.train_step(batch)
        if not result.get("optimizer_step"):
            raise RuntimeError("Qwen benchmark step did not reach optimizer boundary")
        return result

    def residency_objects(self) -> list[Any]:
        return [self.model, self.optimizer, self._last_device_batch]

    def benchmark_metadata(self) -> dict[str, Any]:
        metadata = asdict(self.config)
        for runner_setting in (
            "device", "compile_model", "compile_backend", "optimization_variant"
        ):
            metadata.pop(runner_setting)
        return metadata

    def setup_data(self) -> None:
        """Download and tokenize the benchmark's deterministic Dolly subsets."""

        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.config)
        train, evaluation = load_dolly_splits(self.config)

        def tokenize(example: Mapping[str, Any]) -> dict[str, list[int]]:
            return {
                key: value.tolist()
                for key, value in encode_response_only(
                    example, self.tokenizer, self.config.sequence_length
                ).items()
            }

        columns = train.column_names
        self.train_dataset = train.map(tokenize, remove_columns=columns)
        self.eval_dataset = evaluation.map(tokenize, remove_columns=columns)
        self.train_dataset.set_format("torch")
        self.eval_dataset.set_format("torch")

    def batches(self, *, evaluation: bool = False) -> Iterator[dict[str, Tensor]]:
        """Yield deterministic microbatches of one (the benchmark contract)."""

        dataset = self.eval_dataset if evaluation else self.train_dataset
        if dataset is None:
            self.setup_data()
            dataset = self.eval_dataset if evaluation else self.train_dataset
        assert dataset is not None
        for row in dataset:
            yield {key: value for key, value in row.items()}

    @classmethod
    def from_args(cls, args: Any) -> "QwenSFTWorkload":
        return cls(
            QwenSFTConfig(
                training=args.training,
                device=args.device,
                compile_model=getattr(args, "compile", False),
                optimization_variant=getattr(args, "optimization_variant", "upstream"),
                gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
            )
        )

    def prepare_example(self, example: Mapping[str, Any]) -> dict[str, Tensor]:
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.config)
        return encode_response_only(
            example, self.tokenizer, self.config.sequence_length
        )

    def train_step(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor | bool]:
        """Run one microbatch, stepping AdamW at the accumulation boundary."""

        self.model.train()
        device_batch = {
            key: value.unsqueeze(0).to(self.config.device, non_blocking=True)
            if value.ndim == 1
            else value.to(self.config.device, non_blocking=True)
            for key, value in batch.items()
        }
        self._last_device_batch = device_batch
        autocast = (
            torch.autocast(device_type="xpu", dtype=torch.bfloat16)
            if self.config.device.startswith("xpu")
            else contextlib.nullcontext()
        )
        with autocast:
            output = self.model(**device_batch)
            loss = output.loss / self.config.gradient_accumulation_steps
        loss.backward()
        self._micro_step += 1
        stepped = self._micro_step % self.config.gradient_accumulation_steps == 0
        if stepped:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return {
            # Converting to float synchronizes XPU inside the timed region.
            "loss": loss.detach() * self.config.gradient_accumulation_steps,
            "optimizer_step": stepped,
        }

    def synchronize(self) -> None:
        if self.config.device.startswith("xpu"):
            torch.xpu.synchronize()

    def assert_xpu_residency(self) -> None:
        """Reject accidental CPU model, gradient, or optimizer-state fallback."""

        expected = torch.device(self.config.device).type
        if expected != "xpu":
            raise ValueError("residency validation is meaningful only for XPU runs")
        for name, parameter in self.model.named_parameters():
            if parameter.device.type != expected:
                raise RuntimeError(f"parameter {name} is on {parameter.device}")
            if parameter.grad is not None and parameter.grad.device.type != expected:
                raise RuntimeError(f"gradient {name} is on {parameter.grad.device}")
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, Tensor) and value.device.type != expected:
                    raise RuntimeError(f"optimizer tensor is on {value.device}")

    def save_checkpoint(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model = getattr(self.model, "_orig_mod", self.model)
        if self.config.training == "lora":
            try:
                from peft import get_peft_model_state_dict

                model_state = get_peft_model_state_dict(model)
            except ImportError:
                model_state = {
                    name: tensor
                    for name, tensor in model.state_dict().items()
                    if "lora_" in name
                }
        else:
            model_state = model.state_dict()
        state = {
            "model": _cpu_tree(model_state),
            "optimizer": _cpu_tree(self.optimizer.state_dict()),
            "micro_step": self._micro_step,
        }
        path = directory / "training_state.pt"
        torch.save(state, path)
        (directory / "benchmark_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True) + "\n"
        )
        return path

    def load_checkpoint(self, directory: str | Path) -> None:
        state = torch.load(
            Path(directory) / "training_state.pt",
            map_location=self.config.device,
            weights_only=True,
        )
        model = getattr(self.model, "_orig_mod", self.model)
        if self.config.training == "lora":
            try:
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(model, state["model"])
            except ImportError:
                model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._micro_step = int(state["micro_step"])


def add_cli_arguments(parser: Any) -> None:
    parser.add_argument("--training", choices=("lora", "full"), default="lora")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--compile", action="store_true", dest="compile")
    parser.add_argument(
        "--optimization-variant", choices=("upstream", "ht"), default="upstream"
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")


def create_workload(args: Any) -> QwenSFTWorkload:
    return QwenSFTWorkload.from_args(args)


def create_benchmark_workload(
    variant: str, device: str, options: dict[str, str]
) -> QwenSFTWorkload:
    config = QwenSFTConfig.from_options(
        options, device=device, compile_model=variant in {"xpu-compiled", "ht"}
    )
    if variant == "ht":
        config = QwenSFTConfig(
            **{**asdict(config), "optimization_variant": "ht"}
        )
    return QwenSFTWorkload(config)


register_workload("qwen35", create_benchmark_workload)
