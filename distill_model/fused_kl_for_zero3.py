# -*- coding: utf-8 -*-

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import input_guard, is_amd

try:
    import deepspeed
    from deepspeed.utils import safe_get_local_grad
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

MAX_FUSED_SIZE = 65536 // 2
STATIC_WARPS = 32 if not is_amd else 16


@triton.jit
def kl_div_kernel(
    logits,
    target_logits,
    loss,
    s_logits,
    s_loss,
    reduction: tl.constexpr,
    N: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr
):
    i_n = tl.program_id(0).to(tl.int64)

    logits += i_n * s_logits
    target_logits += i_n * s_logits

    sm = float('-inf')
    tm = float('-inf')
    sd, td = 0.0, 0.0

    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sm = tl.max(b_sl)
        m_new = tl.maximum(sm, b_sm)
        sd = sd * exp(sm - m_new) + tl.sum(exp(b_sl - m_new))
        sm = m_new
        
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tm = tl.max(b_tl)
        m_new = tl.maximum(tm, b_tm)
        td = td * exp(tm - m_new) + tl.sum(exp(b_tl - m_new))
        tm = m_new

    b_loss = 0.
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sp_log = b_sl - sm - log(sd)
        b_tp_log = b_tl - tm - log(td)
        b_sp = exp(b_sp_log)
        b_tp = exp(b_tp_log)
        b_kl = tl.where(o_x < V, b_tp * (b_tp_log - b_sp_log), 0)
        b_dl = -b_tp + b_sp
        b_loss += tl.sum(b_kl)
        if reduction == 'batchmean':
            b_dl = b_dl / N
        tl.store(logits + o_x, b_dl, mask=o_x < V)

    if reduction == 'batchmean':
        b_loss = b_loss / N
    tl.store(loss + i_n * s_loss, b_loss)


@triton.jit
def elementwise_mul_kernel(
    x,
    g,
    N: tl.constexpr,
    B: tl.constexpr
):
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def _accumulate_gradient_to_param(param: torch.Tensor, grad_tensor: torch.Tensor):
    """
    直接将梯度累积到参数的.grad属性中，绕过autograd返回机制
    这样ZeRO-3会自动处理梯度的分片和归约
    """
    if param.grad is None:
        param.grad = torch.zeros_like(param)
    
    if HAS_DEEPSPEED and hasattr(param, 'ds_id'):
        # ZeRO-3参数：使用DeepSpeed的梯度累积机制
        # 在gathered状态下累积完整梯度，DeepSpeed会自动处理分片
        with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.data.add_(grad_tensor)
    else:
        # 常规参数：直接累积
        param.grad.data.add_(grad_tensor)


class FusedKLDivLossFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        target_x: torch.Tensor,
        weight: torch.Tensor,
        target_weight: torch.Tensor,
        reduction: str
    ):
        device = x.device
        
        # 始终使用Triton优化版本，但处理ZeRO-3的梯度累积
        using_zero3 = HAS_DEEPSPEED and hasattr(weight, 'ds_id')
        
        if using_zero3:
            # ZeRO-3: 在gathered状态下计算，但梯度直接累积到参数
            with deepspeed.zero.GatheredParameters([weight, target_weight], modifier_rank=None):
                gathered_weight = weight
                gathered_target_weight = target_weight
                
                # 使用Triton优化计算
                loss, dx, dw = _compute_fused_kl_div(
                    x, target_x, gathered_weight, gathered_target_weight, reduction
                )
                
                # 关键：直接累积梯度到参数，不通过autograd返回
                if dw is not None and weight.requires_grad:
                    _accumulate_gradient_to_param(weight, dw)
                
                # 只保存输入梯度用于backward
                ctx.save_for_backward(dx)
                ctx.using_zero3 = True
                
        else:
            # 非ZeRO-3: 正常的Triton计算
            loss, dx, dw = _compute_fused_kl_div(
                x, target_x, weight, target_weight, reduction
            )
            ctx.save_for_backward(dx, dw)
            ctx.using_zero3 = False
            
        return loss

    @staticmethod
    @input_guard
    def backward(ctx, grad_output):
        if ctx.using_zero3:
            # ZeRO-3: 只返回输入梯度，权重梯度已经直接累积了
            dx, = ctx.saved_tensors
            
            if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
                N, H = dx.shape
                B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

                elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
                    x=dx,
                    g=grad_output,
                    N=N*H,
                    B=B,
                    num_warps=STATIC_WARPS,
                )
            
            # 权重梯度返回None，因为已经直接累积到参数了
            return dx, None, None, None, None
            
        else:
            # 非ZeRO-3: 正常处理
            dx, dw = ctx.saved_tensors
            
            if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
                N, H = dx.shape
                B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

                elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
                    x=dx,
                    g=grad_output,
                    N=N*H,
                    B=B,
                    num_warps=STATIC_WARPS,
                )

                if dw is not None:
                    V, H = dw.shape
                    elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                        x=dw,
                        g=grad_output,
                        N=V*H,
                        B=B,
                        num_warps=STATIC_WARPS,
                    )
            
            return dx, None, dw, None, None


def _compute_fused_kl_div(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = 'batchmean'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """核心Triton计算逻辑"""
    device = x.device
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    NC = min(8, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    dx = torch.zeros_like(x, device=device)
    dw = torch.zeros_like(weight, device=device) if weight is not None else None
    loss = torch.zeros(N, dtype=torch.float32, device=device)

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_sx = x[start:end]
        c_tx = target_x[start:end]
        
        c_sl = F.linear(c_sx, weight)
        c_tl = F.linear(c_tx, target_weight)
        c_loss = loss[start:end]

        kl_div_kernel[(c_sx.shape[0],)](
            logits=c_sl,
            target_logits=c_tl,
            loss=c_loss,
            s_logits=c_sl.stride(-2),
            s_loss=c_loss.stride(-1),
            reduction=reduction,
            N=N,
            V=V,
            BV=BV,
            num_warps=STATIC_WARPS
        )

        dx[start:end] = torch.mm(c_sl, weight)
        
        if dw is not None:
            torch.addmm(input=dw, mat1=c_sl.t(), mat2=c_sx, out=dw)

    return loss.sum(), dx, dw


def fused_kl_div_loss_for_zero3(
    x: torch.Tensor,
    target_x: torch.Tensor,
    weight: torch.Tensor,
    target_weight: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    内存高效的ZeRO-3兼容融合KL散度损失函数。
    
    在所有情况下都使用Triton优化，通过直接梯度累积来解决ZeRO-3兼容性问题。
    """
    return FusedKLDivLossFunction.apply(
        x, target_x, weight, target_weight, reduction
    )


class FusedKLDivLoss(nn.Module):
    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()
        assert reduction in ['batchmean'], f"reduction: {reduction} is not supported"
        self.reduction = reduction

    def forward(self, x, target_x, weight, target_weight):
        return fused_kl_div_loss(x, target_x, weight, target_weight, self.reduction)
