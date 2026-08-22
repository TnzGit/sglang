# Adapted from CompressedTensorsW4A4Fp4 (vLLM-derived) for pre-sm80 devices.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 checkpoint served through Marlin dequant-in-kernel on pre-sm80.

The stock CompressedTensorsW4A4Fp4 scheme runs FP4 activations on native
FP4 tensor cores (SM100+). Turing and other pre-sm80 devices have no FP4
hardware path, but the Marlin kernel family dequantizes FP4 weights to
fp16/bf16 in-register and computes with fp16 MMA — validated on RTX 2080 Ti
(rel err 7e-4 vs dequant reference) via the local-vLLM bridge.

Checkpoint layout (compressed-tensors NVFP4):
  weight_packed       uint8  [N, K/2]   two fp4 nibbles per byte
  weight_scale        e4m3   [N, K/16]  per-16-channel block scales
  weight_global_scale fp32/half         per-tensor global scale
Activations are consumed as bf16/fp16 directly; the checkpoint's FP4
activation-quant metadata is intentionally ignored here.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)

__all__ = ["CompressedTensorsW4A4Fp4Marlin"]


class CompressedTensorsW4A4Fp4Marlin(CompressedTensorsLinearScheme):
    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        input_size_per_partition_list: list[int] | None = None,
        params_dtype: torch.dtype = torch.float16,
        weight_loader: Callable | None = None,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # Same parameter names/shapes as CompressedTensorsW4A4Fp4 so the
        # checkpoint weight_loader maps NVFP4 tensors identically.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # Kept only so the standard CT weight loader can consume the
        # checkpoint's input-global-scale tensor; unused at runtime.
        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        # Collapse per-shard global scales to one scalar (max), mirroring the
        # stock scheme's semantics.
        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )
        if hasattr(layer, "input_global_scale"):
            delattr(layer, "input_global_scale")

        # prepare_nvfp4_layer_for_marlin reads these attributes.
        layer.weight = torch.nn.Parameter(
            layer.weight_packed.data, requires_grad=False
        )
        if getattr(layer, "quant_config", None) is None or not hasattr(
            layer.quant_config, "group_size"
        ):
            layer.quant_config = SimpleNamespace(group_size=self.group_size)

        prepare_nvfp4_layer_for_marlin(layer)

        # prepare leaves workspace on the layer; release loader-format aliases.
        if hasattr(layer, "weight_packed"):
            delattr(layer, "weight_packed")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            x,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.workspace,
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            bias=bias,
            use_fp32_reduce=True,
        )
