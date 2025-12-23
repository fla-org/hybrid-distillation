from __future__ import annotations
import torch
import torch.nn as nn
from transformers import Trainer
import torch.nn.functional as F
from fla.modules.fused_kl_div import FusedKLDivLoss

__all__ = ["DistillTrainer", "FinetuneTrainer", "KDTrainer"]

class _BaseTrainer(Trainer):
    """
    A thin wrapper around Huggingface Trainer that only overrides compute_loss().
    Everything else (DDP, ZeRO, gradient‐accumulation, mixed precision,
    checkpoint‑rotation, etc.) is delegated to Hugging Face / DeepSpeed / Accelerate.
    """
    def __init__(self, *args,
                 mse_factor: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_factor = mse_factor
        self.criterion_ce  = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_mse = nn.MSELoss(reduction="mean")

class DistillTrainer(_BaseTrainer):
    """
    Stage‑1 trainer used by rapid‑distill / LoLCats‑AT etc.
    Assumes the model.forward(..) returns a tuple of attentions where
    `attn[layer][0]` is the teacher map and `attn[layer][1]` is the student map.
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):

        inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "labels"}
        outputs = model(**inputs)

        # Gather distillation losses that were stashed
        #    by every AttentionDistillationWrapper
        per_layer_losses = []
        for layer in model.model.layers:
            sa = layer.attn
            if hasattr(sa, "distill_loss"):
                per_layer_losses.append(sa.distill_loss)

        if per_layer_losses:                       # stack → mean
            loss = torch.stack(per_layer_losses).mean() * self.mse_factor
        else:                                      # should never happen, but be safe
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)

        if return_outputs:
            extra = {"loss_mse": loss.detach().cpu().item(),
                     "mse_factor": self.mse_factor}
            return loss, {**outputs, **extra}
        return loss

class FinetuneTrainer(_BaseTrainer):
    """
    LM fine‑tuning stage – classic causal‑LM cross‑entropy.
    We compute loss manually so it stays identical to your current code
    (shift‑inputs‑by‑1 and ignore padding).
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_keys = {"input_ids"}
        data = {k: v.to(model.device) for k, v in inputs.items() if k in input_keys}

        logits = model(**data, use_cache=False).logits
        logits  = logits[:, :-1, :].contiguous()
        targets = inputs['input_ids'][:, 1:].contiguous()

        loss = self.criterion_ce(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        if return_outputs:
            return (loss,
                    {"ppl": torch.exp(loss).item(),
                     "seq_len": targets.size(1)+1,
                     "logits": logits})
        return loss


class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        kl_weight=1.0,
        ce_weight=1.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight

        # teacher_model can be large: put in eval mode, possibly wrap in deepspeed
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # using FLA's memory-efficient chunked KL divergence loss
        loss = model.forward_kl(teacher=self.teacher_model, input_ids=inputs["input_ids"])
        return (loss, None) if return_outputs else loss