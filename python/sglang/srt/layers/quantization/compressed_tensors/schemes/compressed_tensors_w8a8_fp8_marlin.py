# SPDX-License-Identifier: Apache-2.0
"""W8A8-FP8 checkpoint layers served through Marlin on pre-sm80 devices.

The stock CompressedTensorsW8A8Fp8 scheme requires FP8 activation hardware
(SM89+). Pre-sm80 devices have no FP8 path, but the Marlin kernel family
dequantizes FP8 weights to fp16 in-register and computes with fp16 MMA —
validated on RTX 2080 Ti (rel err 1e-3 vs reference) via the local-vLLM
bridge.

Checkpoint layout (compressed-tensors FP8, channel strategy):
  weight        e4m3   [N, K]
  weight_scale  f32    [N, 1]   per-output-channel scale
Activations are consumed as bf16/fp16 directly; the checkpoint's dynamic
FP8 activation-quant metadata is intentionally ignored here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.kernels.ops.quantization._turing_marlin_bridge import (
    pre_sm80,
    turing_fp8_marlin_gemm,
    turing_prepare_fp8_weight_for_marlin,
)

__all__ = ["CompressedTensorsW8A8Fp8Marlin"]


class CompressedTensorsW8A8Fp8Marlin(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant):
        if weight_quant.strategy not in (
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.TENSOR,
        ):
            raise NotImplementedError(
                "W8A8Fp8Marlin supports channel/tensor FP8 weight scales; "
                f"got strategy={weight_quant.strategy}. Block-wise FP8 is a "
                "DeepSeek-style format not targeted by the Turing port."
            )
        self.strategy = weight_quant.strategy

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype
        layer.params_dtype = params_dtype

        # Same parameter names/shapes as CompressedTensorsW8A8Fp8 so the CT
        # weight loader maps checkpoint tensors unchanged.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition, 1), dtype=torch.float32
                ),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:  # TENSOR
            weight_scale = torch.nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32, device="cuda"),
                requires_grad=False,
            )
            layer.register_parameter("weight_scale", weight_scale)
            return

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        # Dequant to the activation dtype, then re-quantize into Marlin's
        # per-row FP8 layout. The stored channel scales are exact multipliers,
        # so this round-trip is lossless for checkpoints following the
        # scale=max/448 convention.
        # Dimensions come from the packed tensor itself: RowParallel layers
        # (o_proj/down_proj) don't expose output_size_per_partition.
        device = layer.weight.device
        size_n, size_k = layer.weight.shape

        w = layer.weight.to(layer.orig_dtype)
        scale = layer.weight_scale.to(device)
        if scale.dim() == 2:
            w_ref = w * scale.view(-1, 1).to(w.dtype)
        else:
            w_ref = w * scale.to(w.dtype)
        w_ref = w_ref.reshape(size_n, size_k).to(torch.float16)

        marlin_qweight, marlin_scales = turing_prepare_fp8_weight_for_marlin(w_ref)

        layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            marlin_scales, requires_grad=False
        )
        layer.workspace = torch.zeros(
            max(4096, size_n // 16 + 64),
            dtype=torch.int32,
            device=device,
        )
        # Marlin-repacked shapes differ from the logical ones when padding
        # was applied; keep both for apply().
        layer.marlin_size_n = size_n
        layer.marlin_size_k = size_k

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = turing_fp8_marlin_gemm(
            x,
            None,
            layer.weight,
            layer.weight_scale,
            layer.workspace,
            x.shape[0],
            layer.marlin_size_n,
            layer.marlin_size_k,
        )
        if bias is not None:
            out = out + bias
        return out
