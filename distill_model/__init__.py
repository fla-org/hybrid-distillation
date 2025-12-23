from .config_distilled_student import StudentConfig
from .modeling_distilled_student import StudentModel, StudentForCausalLM, get_student_attention_class
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

