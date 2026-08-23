"""Turing (pre-sm80) bridge for Marlin kernels.

sglang's JIT gptq-marlin kernel targets sm80+ only: the multi-type Turing
MMA path (marlin_mma.h, m8n8k16 fragments) present in upstream vLLM has not
been ported into the JIT tree yet. On cc <= 7 devices we delegate to the
locally installed vLLM build, whose compiled marlin kernels cover sm75
(validated on RTX 2080 Ti: 65/66 kernel configs pass; the only failure is
the bfloat16 type, which pre-sm80 devices cannot run anyway).

ScalarType ids are identical between both packages (common upstream
lineage), so only a from_id conversion happens at the boundary.

This bridge is a Phase-2 unblock for the turing-sm75 port; the long-term
fix is porting the Turing MMA sections into sglang's JIT marlin sources.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def pre_sm80() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] < 8


@lru_cache(maxsize=1)
def _vllm_ops():
    try:
        import vllm._custom_ops as ops  # noqa: PLC0415

        return ops
    except Exception:  # pragma: no cover - environment without vLLM
        return None


def require_vllm_ops():
    ops = _vllm_ops()
    if ops is None:
        raise RuntimeError(
            "Turing (cc <= 7) Marlin support requires the locally installed "
            "vLLM package (its compiled marlin kernels cover sm75). Install "
            "vLLM or run on sm80+ where sglang's JIT marlin is used."
        )
    return ops


def _vllm_scalar_type(b_q_type):
    import vllm.scalar_type as vst  # noqa: PLC0415

    return vst.ScalarType.from_id(b_q_type.id)


def turing_marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """gptq_marlin_gemm bridge: sglang signature -> vLLM marlin_gemm."""
    ops = require_vllm_ops()
    if c is None:
        c = torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)
    # Turing marlin only accepts fp16/int8 activations. The bf16 draft path
    # (pre-sm80 DFlash2) produces bf16 inputs; every linear's input is a
    # post-norm stream (small magnitude), so a fp16 round-trip is safe here.
    out_dtype = None
    if a.dtype == torch.bfloat16:
        out_dtype = c.dtype
        a = a.to(torch.float16)
        c = torch.empty((size_m, size_n), dtype=torch.float16, device=a.device)
    # vLLM's compiled kernel requires the NVFP4 global scale in fp32; sglang
    # helpers may keep it in the activation dtype.
    if (
        global_scale is not None
        and global_scale.numel() > 0
        and global_scale.dtype != torch.float32
    ):
        global_scale = global_scale.to(torch.float32)
    out = ops.marlin_gemm(
        a,
        c,
        b_q_weight,
        None,  # b_bias (sglang GPTQ/AWQ paths carry bias separately)
        b_scales,
        None,  # a_scales (fp8 activations only; not on Turing)
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        _vllm_scalar_type(b_q_type),
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
    if out_dtype is not None:
        out = out.to(out_dtype)
        c = out
    return out


def turing_gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """gptq_marlin_repack bridge."""
    ops = require_vllm_ops()
    return ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)


def turing_awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """awq_marlin_repack bridge (AWQ zero-point layout -> marlin)."""
    ops = require_vllm_ops()
    return ops.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)


def turing_prepare_fp8_weight_for_marlin(
    weight_f16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an fp16 [N, K] weight to Marlin's FP8-weight layout.

    Uses per-output-row FP8 scales (group_size=-1), matching the
    compressed-tensors "channel" strategy of W8A8-FP8 checkpoints.
    Returns (marlin_qweight int32, marlin_scales).
    """
    ops = require_vllm_ops()
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        marlin_quant_fp8_torch,  # noqa: PLC0415
    )

    _, marlin_qweight, marlin_scales = marlin_quant_fp8_torch(
        weight_f16, group_size=-1, input_dtype=None
    )
    return marlin_qweight, marlin_scales


def turing_fp8_marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
    use_fp32_reduce: bool = True,
) -> torch.Tensor:
    """FP8-weight GEMM bridge: bf16/fp16 activations against e4m3 weights."""
    import vllm.scalar_type as vst  # noqa: PLC0415

    ops = require_vllm_ops()
    if c is None:
        c = torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)
    return ops.marlin_gemm(
        a,
        c,
        b_q_weight,
        None,  # b_bias
        b_scales,
        None,  # a_scales (activations stay bf16/fp16 on this path)
        None,  # global_scale (weights are pre-scaled at repack time)
        None,  # b_zeros
        None,  # g_idx
        None,  # perm
        workspace,
        vst.scalar_types.float8_e4m3fn,
        size_m,
        size_n,
        size_k,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )
