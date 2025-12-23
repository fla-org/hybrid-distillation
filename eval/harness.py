# -*- coding: utf-8 -*-

from __future__ import annotations

import fla
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentModel, StudentForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

@register_model('fla')
class FlashLinearAttentionLMWrapper(HFLM):
    def __init__(self, **kwargs) -> FlashLinearAttentionLMWrapper:
        # TODO: provide options for doing inference with different kernels
        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
