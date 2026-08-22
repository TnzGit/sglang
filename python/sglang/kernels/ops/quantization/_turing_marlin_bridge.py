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
    return ops.marlin_gemm(
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
