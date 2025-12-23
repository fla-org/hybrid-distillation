import torch
import torch.nn as nn

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)        
        self.student_attn = student_cls(config, layer_idx)

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)

        self.student_attn = student_cls(config, layer_idx)
        self.student_attn.init_from_teacher(self.teacher_attn)
        self.distill_loss = torch.tensor(0.0)

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        kwargs["use_cache"] = False # Disable cache for teacher pass during training

        # Teacher pass – no gradients
        with torch.no_grad():
            # CORRECTED: Unpack 3 values from the standard attention layer
            t_hidden, _, _ = self.teacher_attn(*args, **kwargs)

        # Student pass (this is what must flow back to the decoder)
        s_hidden, _, _ = self.student_attn(*args, **kwargs)

        # Stash distillation loss for the caller to consume
        self.distill_loss = torch.linalg.vector_norm(
            t_hidden - s_hidden, dim=-1
        ).mean() * (t_hidden.size(-1) ** -0.5)

        return t_hidden, None, None
