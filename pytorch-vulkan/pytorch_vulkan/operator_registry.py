from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ShaderVariant:
    file: str
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class OperatorSpec:
    schema: str
    native_key: str
    shaders: Mapping[str, ShaderVariant]
    layouts: tuple[str, ...] = ("contiguous",)
    ranks: tuple[int, ...] = (1, 2, 3, 4)
    fx_names: tuple[str, ...] = ()
    dispatch: str = "pointwise"
    inputs: int = 1
    outputs: int = 1
    bindings: int | None = None
    push_constants: tuple[str, ...] = ("num_elements:u32",)
    autograd: str = "composite"
    autoload: bool = True
    tolerances: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "float32": (1e-5, 1e-5),
            "float16": (2e-3, 2e-3),
            "bfloat16": (2e-2, 2e-2),
        }
    )


def _variant(file: str, *capabilities: str) -> ShaderVariant:
    return ShaderVariant(file, frozenset(capabilities))


FP16_CAPABILITIES = ("shader_float16", "storage_buffer16_bit_access")


def _pointwise(
    schema: str,
    native_key: str,
    *,
    inputs: int,
    fx_names: tuple[str, ...] = (),
    f16: bool = True,
    bf16: bool = False,
    autograd: str = "composite",
    push_constants: tuple[str, ...] = ("num_elements:u32",),
    bindings: int | None = None,
) -> OperatorSpec:
    shaders = {"float32": _variant(f"{native_key}.comp")}
    if f16:
        shaders["float16"] = _variant(
            f"{native_key}_f16.comp", *FP16_CAPABILITIES
        )
    if bf16:
        shaders["bfloat16"] = _variant(f"{native_key}_bf16.comp")
    return OperatorSpec(
        schema=schema,
        native_key=native_key,
        shaders=shaders,
        fx_names=fx_names,
        inputs=inputs,
        autograd=autograd,
        push_constants=push_constants,
        bindings=bindings,
    )


OPERATOR_REGISTRY: tuple[OperatorSpec, ...] = (
    _pointwise(
        "aten::add.Tensor",
        "add",
        inputs=2,
        fx_names=("add",),
        bf16=True,
    ),
    _pointwise(
        "aten::sub.Tensor",
        "sub",
        inputs=2,
        fx_names=("sub",),
        bf16=True,
    ),
    _pointwise(
        "aten::mul.Tensor",
        "mul",
        inputs=2,
        fx_names=("mul",),
        bf16=True,
    ),
    _pointwise("aten::div.Tensor", "div", inputs=2, bf16=True),
    _pointwise("aten::relu", "relu", inputs=1, fx_names=("relu",)),
    _pointwise("aten::neg", "neg", inputs=1, fx_names=("neg",)),
    _pointwise("aten::sigmoid", "sigmoid", inputs=1, fx_names=("sigmoid",)),
    _pointwise("aten::tanh", "tanh", inputs=1, fx_names=("tanh",)),
    _pointwise("aten::exp", "exp", inputs=1, fx_names=("exp",)),
    _pointwise("aten::log", "log", inputs=1, fx_names=("log",)),
    _pointwise("aten::silu", "silu", inputs=1, bf16=True),
    _pointwise("aten::sqrt", "sqrt", inputs=1, f16=False),
    _pointwise("aten::abs", "abs", inputs=1, f16=False),
    _pointwise("aten::rsqrt", "rsqrt", inputs=1, f16=False),
    _pointwise(
        "aten::pow.Tensor_Scalar",
        "pow",
        inputs=1,
        push_constants=("num_elements:u32", "exponent:f32"),
    ),
    _pointwise(
        "aten::add_.Scalar",
        "add_scalar",
        inputs=1,
        bf16=True,
        push_constants=("num_elements:u32", "value:f32"),
        bindings=1,
    ),
    _pointwise(
        "aten::mul_.Scalar",
        "mul_scalar",
        inputs=1,
        bf16=True,
        push_constants=("num_elements:u32", "value:f32"),
        bindings=1,
    ),
    _pointwise(
        "aten::div_.Scalar",
        "div_scalar",
        inputs=1,
        bf16=True,
        push_constants=("num_elements:u32", "value:f32"),
        bindings=1,
    ),
    _pointwise(
        "aten::fill_.Scalar",
        "fill_scalar",
        inputs=1,
        bf16=True,
        push_constants=("num_elements:u32", "value:f32"),
        bindings=1,
    ),
    _pointwise(
        "pytorch_vulkan::causal_mask_",
        "causal_mask",
        inputs=1,
        fx_names=(),
        bf16=False,
        autograd="none",
        push_constants=(
            "num_elements:u32",
            "query_length:u32",
            "key_length:u32",
        ),
        bindings=1,
    ),
    OperatorSpec(
        schema="pytorch_vulkan::boolean_mask_",
        native_key="boolean_mask",
        shaders={
            "float32": _variant("boolean_mask.comp"),
            "float16": _variant(
                "boolean_mask_f16.comp", *FP16_CAPABILITIES
            ),
        },
        layouts=("strided",),
        ranks=(1, 2, 3, 4),
        inputs=2,
        autograd="none",
        bindings=2,
        push_constants=(
            "num_elements:u32",
            "ndim:u32",
            "size0:u32",
            "size1:u32",
            "size2:u32",
            "size3:u32",
            "mask_size0:u32",
            "mask_size1:u32",
            "mask_size2:u32",
            "mask_size3:u32",
            "mask_stride0:u32",
            "mask_stride1:u32",
            "mask_stride2:u32",
            "mask_stride3:u32",
            "mask_offset:u32",
        ),
    ),
    OperatorSpec(
        schema="pytorch_vulkan::sdpa_nan_to_zero_",
        native_key="sdpa_nan_to_zero",
        shaders={
            "float32": _variant("sdpa_nan_to_zero.comp"),
            "float16": _variant(
                "sdpa_nan_to_zero_f16.comp", *FP16_CAPABILITIES
            ),
            "bfloat16": _variant("sdpa_nan_to_zero_bf16.comp"),
        },
        inputs=1,
        outputs=0,
        autograd="none",
    ),
    _pointwise(
        "aten::addcdiv",
        "addcdiv",
        inputs=3,
        push_constants=("num_elements:u32", "value:f32"),
    ),
    _pointwise(
        "aten::addcmul",
        "addcmul",
        inputs=3,
        push_constants=("num_elements:u32", "value:f32"),
    ),
    _pointwise(
        "aten::lerp.Scalar",
        "lerp",
        inputs=2,
        push_constants=("num_elements:u32", "weight:f32"),
    ),
    _pointwise("aten::lerp.Tensor", "lerp_tensor", inputs=3),
    _pointwise(
        "aten::threshold_backward",
        "threshold_backward",
        inputs=2,
        fx_names=("threshold_backward",),
        autograd="backward",
        push_constants=("num_elements:u32", "threshold:f32"),
    ),
    OperatorSpec(
        schema="aten::copy_",
        native_key="copy",
        shaders={
            "float32": _variant("copy.comp"),
            "float16": _variant("copy_f16.comp", *FP16_CAPABILITIES),
            "bfloat16": _variant("copy_bf16.comp"),
            "float16_to_float32": _variant(
                "copy_f16_to_f32.comp", *FP16_CAPABILITIES
            ),
            "float32_to_float16": _variant(
                "copy_f32_to_f16.comp", *FP16_CAPABILITIES
            ),
            "bfloat16_to_float32": _variant("copy_bf16_to_f32.comp"),
            "float32_to_bfloat16": _variant("copy_f32_to_bf16.comp"),
            "int64_to_float32": _variant("copy_i64_to_f32.comp"),
        },
        layouts=("strided",),
        ranks=(0, 1, 2, 3, 4),
        inputs=1,
        autograd="none",
        push_constants=(
            "num_elements:u32",
            "ndim:u32",
            "size0:u32",
            "size1:u32",
            "size2:u32",
            "size3:u32",
            "src_stride0:u32",
            "src_stride1:u32",
            "src_stride2:u32",
            "src_stride3:u32",
            "src_offset:u32",
            "dst_stride0:u32",
            "dst_stride1:u32",
            "dst_stride2:u32",
            "dst_stride3:u32",
            "dst_offset:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::cat",
        native_key="cat2",
        shaders={
            "float32": _variant("cat2.comp"),
            "float16": _variant("cat2_f16.comp", *FP16_CAPABILITIES),
        },
        inputs=2,
        push_constants=(
            "num_elements:u32",
            "inner_a:u32",
            "inner_b:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::embedding",
        native_key="embedding",
        shaders={
            "float32": _variant("embedding.comp"),
            "float16": _variant("embedding_f16.comp", *FP16_CAPABILITIES),
        },
        ranks=(2,),
        inputs=2,
        push_constants=(
            "num_indices:u32",
            "embedding_dim:u32",
            "num_weights:u32",
            "index_words:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::embedding_dense_backward",
        native_key="embedding_backward",
        shaders={
            "float32": _variant("embedding_backward.comp"),
            "float16": _variant(
                "embedding_backward_f16.comp", *FP16_CAPABILITIES
            ),
        },
        ranks=(2,),
        inputs=2,
        autograd="backward",
        push_constants=(
            "num_indices:u32",
            "num_weights:u32",
            "embedding_dim:u32",
            "padding_idx:i32",
            "scale_grad_by_freq:u32",
            "index_words:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::one_hot",
        native_key="one_hot",
        shaders={"int64": _variant("one_hot.comp")},
        layouts=("strided",),
        ranks=(0, 1, 2, 3, 4),
        inputs=1,
        autograd="none",
        push_constants=(
            "num_elements:u32",
            "num_classes:u32",
            "ndim:u32",
            "size0:u32",
            "size1:u32",
            "size2:u32",
            "size3:u32",
            "stride0:u32",
            "stride1:u32",
            "stride2:u32",
            "stride3:u32",
            "storage_offset:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::mm",
        native_key="matmul_tiled",
        shaders={
            "float32": _variant("matmul_tiled.comp"),
            "float16": _variant(
                "matmul_tiled_f16.comp", *FP16_CAPABILITIES
            ),
        },
        ranks=(2,),
        fx_names=("mm",),
        dispatch="mm",
        inputs=2,
        push_constants=("m:u32", "n:u32", "k:u32"),
        tolerances={
            "float32": (2e-4, 2e-4),
            "float16": (1e-2, 1e-2),
        },
    ),
    OperatorSpec(
        schema="aten::bmm",
        native_key="bmm",
        shaders={
            "float32": _variant("bmm.comp"),
            "float16": _variant("bmm_f16.comp", *FP16_CAPABILITIES),
            "cooperative_float16": _variant(
                "bmm_coopmat.comp",
                *FP16_CAPABILITIES,
                "cooperative_matrix_nv",
            ),
        },
        ranks=(3,),
        fx_names=("bmm",),
        dispatch="bmm",
        inputs=2,
        push_constants=("batch:u32", "m:u32", "n:u32", "k:u32"),
    ),
    OperatorSpec(
        schema="aten::_softmax",
        native_key="softmax",
        shaders={
            "float32": _variant("softmax.comp"),
            "float16": _variant("softmax_f16.comp", *FP16_CAPABILITIES),
            "bfloat16": _variant("softmax_bf16.comp"),
        },
        fx_names=("_softmax",),
        dispatch="softmax",
        inputs=1,
        push_constants=("outer_size:u32", "dim_size:u32"),
    ),
    OperatorSpec(
        schema="aten::_softmax_backward_data",
        native_key="softmax_backward",
        shaders={"float32": _variant("softmax_backward.comp")},
        fx_names=("_softmax_backward_data",),
        dispatch="softmax_backward",
        inputs=2,
        autograd="backward",
        push_constants=("outer_size:u32", "dim_size:u32"),
    ),
    OperatorSpec(
        schema="aten::sum",
        native_key="sum",
        shaders={"float32": _variant("sum.comp")},
        dispatch="reduction",
        inputs=1,
        fx_names=("sum",),
    ),
    OperatorSpec(
        schema="aten::sum.dim_IntList",
        native_key="sum_dim",
        shaders={"float32": _variant("sum_dim.comp")},
        dispatch="dim_reduction",
        inputs=1,
        push_constants=(
            "outer_size:u32",
            "reduce_size:u32",
            "inner_size:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::mean",
        native_key="mean",
        shaders={"float32": _variant("mean.comp")},
        dispatch="reduction",
        inputs=1,
        fx_names=("mean",),
    ),
    OperatorSpec(
        schema="aten::mean.dim",
        native_key="mean_dim",
        shaders={"float32": _variant("mean_dim.comp")},
        dispatch="dim_reduction",
        inputs=1,
        push_constants=(
            "outer_size:u32",
            "reduce_size:u32",
            "inner_size:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::transpose.int",
        native_key="transpose",
        shaders={"float32": _variant("transpose.comp")},
        ranks=(2,),
        fx_names=("t", "transpose"),
        dispatch="transpose",
        autoload=False,
        push_constants=("rows:u32", "columns:u32"),
    ),
    OperatorSpec(
        schema="aten::gelu[approximate=tanh]",
        native_key="gelu",
        shaders={
            "float32": _variant("gelu.comp"),
            "float16": _variant("gelu_f16.comp", *FP16_CAPABILITIES),
        },
    ),
    OperatorSpec(
        schema="aten::native_layer_norm",
        native_key="layer_norm",
        shaders={
            "float32": _variant("layer_norm.comp"),
            "float16": _variant("layer_norm_f16.comp", *FP16_CAPABILITIES),
        },
        inputs=3,
        outputs=3,
        bindings=6,
        push_constants=(
            "outer_size:u32",
            "dim_size:u32",
            "eps:f32",
            "has_weight:u32",
            "has_bias:u32",
        ),
    ),
    OperatorSpec(
        schema="aten::_fused_rms_norm",
        native_key="rmsnorm",
        shaders={
            "float32": _variant("rmsnorm.comp"),
            "float16": _variant("rmsnorm_f16.comp", *FP16_CAPABILITIES),
        },
        inputs=2,
        outputs=2,
        bindings=4,
        push_constants=(
            "outer_size:u32",
            "dim_size:u32",
            "eps:f32",
            "has_weight:u32",
        ),
    ),
    OperatorSpec(
        schema="pytorch_vulkan::fused_sdpa",
        native_key="fused_sdpa",
        shaders={"float32": _variant("sdpa.comp")},
        ranks=(4,),
        inputs=3,
        autograd="custom",
        autoload=False,
        push_constants=(
            "sequence:u32",
            "query_dim:u32",
            "value_dim:u32",
            "scale:f32",
        ),
    ),
    OperatorSpec(
        schema="pytorch_vulkan::flash_attention",
        native_key="flash_attention_forward",
        shaders={
            "float32": _variant("flash_attn_fwd_v2.comp"),
            "float16": _variant(
                "flash_attn_fwd_v2_f16.comp", *FP16_CAPABILITIES
            ),
        },
        ranks=(4,),
        inputs=3,
        outputs=2,
        bindings=5,
        autograd="custom",
        autoload=False,
        push_constants=(
            "sequence:u32",
            "query_dim:u32",
            "value_dim:u32",
            "scale:f32",
        ),
    ),
    OperatorSpec(
        schema="pytorch_vulkan::flash_attention_backward",
        native_key="flash_attention_backward",
        shaders={
            "float32": _variant(
                "flash_attn_bwd.comp",
                "shader_buffer_float32_atomic_add",
            ),
            "float16": _variant(
                "flash_attn_bwd_f16.comp",
                *FP16_CAPABILITIES,
                "shader_buffer_float32_atomic_add",
            ),
        },
        ranks=(4,),
        inputs=6,
        outputs=3,
        bindings=9,
        autograd="backward",
        autoload=False,
        push_constants=(
            "sequence:u32",
            "query_dim:u32",
            "value_dim:u32",
            "scale:f32",
        ),
    ),
    OperatorSpec(
        schema="pytorch_vulkan::flash_attention_kvcache",
        native_key="flash_attention_kvcache",
        shaders={
            "float16": _variant(
                "flash_attn_fwd_v2_kvcache_f16.comp",
                *FP16_CAPABILITIES,
            )
        },
        ranks=(4,),
        inputs=3,
        outputs=2,
        bindings=5,
        autograd="none",
        autoload=False,
        push_constants=(
            "query_sequence:u32",
            "kv_sequence:u32",
            "query_dim:u32",
            "value_dim:u32",
            "scale:f32",
        ),
    ),
)


def native_shaders(
    capabilities: Mapping[str, bool],
) -> Iterable[tuple[str, ShaderVariant]]:
    for operator in OPERATOR_REGISTRY:
        if not operator.autoload:
            continue
        for dtype, variant in operator.shaders.items():
            if all(capabilities.get(name, False) for name in variant.capabilities):
                suffix = {
                    "float32": "",
                    "float16": "_f16",
                    "bfloat16": "_bf16",
                    "cooperative_float16": "_coopmat",
                }.get(dtype)
                if suffix is None:
                    key = variant.file.removesuffix(".comp")
                else:
                    key = operator.native_key + suffix
                yield key, variant


def shader_variant(native_key: str, dtype: str) -> ShaderVariant:
    operator = native_operators()[native_key]
    return operator.shaders[dtype]


def fx_operators() -> dict[str, OperatorSpec]:
    return {
        fx_name: operator
        for operator in OPERATOR_REGISTRY
        for fx_name in operator.fx_names
    }


def native_operators() -> dict[str, OperatorSpec]:
    return {operator.native_key: operator for operator in OPERATOR_REGISTRY}


def shader_binding_count(shader: ShaderVariant | str) -> int:
    file = shader.file if isinstance(shader, ShaderVariant) else shader
    matches = {
        operator.bindings
        if operator.bindings is not None
        else operator.inputs + operator.outputs
        for operator in OPERATOR_REGISTRY
        for variant in operator.shaders.values()
        if variant.file == file
    }
    if len(matches) != 1:
        raise KeyError(f"shader {file!r} has no unique descriptor binding count")
    return matches.pop()
