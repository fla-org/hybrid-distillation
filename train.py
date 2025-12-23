import argparse, os, yaml, math, torch
import json
import deepspeed

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from omegaconf import OmegaConf
from training.utils import count_model_params
from hf_trainer import DistillTrainer, FinetuneTrainer, KDTrainer
from accelerate import init_empty_weights
from wrapper import AttentionDistillationWrapper
from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM, get_student_attention_class

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)
import subprocess

import sys
import logging
import torch.distributed as dist
import gc
from safetensors.torch import load_file

local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["TRITON_CACHE_DIR"] = f"triton_cache/{local_rank}"

def get_logger(name: str = None) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if 'RANK' in os.environ and int(os.environ['RANK']) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger
logger = get_logger(__name__)


# workaround for pytorch2.6
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def json_serializer(obj):
    """
    A custom serializer for objects that are not serializable by default json code.
    Specifically, this handles torch.dtype and numpy float types.
    """
    if isinstance(obj, torch.dtype):
        return str(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _prepare_teacher_deepspeed(teacher_model, ds_config_path):
    """
    Wrap a large teacher model in a separate DeepSpeed engine so it can be
    sharded under ZeRO-3. Only needed if your teacher is huge and you want
    DS to manage it. Otherwise you can skip or adapt as needed.
    """
    with open(ds_config_path) as f:
        ds_cfg = json.load(f)

    # Teacher does not need grads
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Force ZeRO-3
    ds_cfg["zero_optimization"]["stage"] = 3

    # Optionally tune bucket sizes
    hidden_size = getattr(teacher_model.config, "hidden_size", None)
    if hidden_size is None and getattr(teacher_model.config, "hidden_sizes", None):
        hidden_size = max(teacher_model.config.hidden_sizes)

    if hidden_size is not None and ds_cfg["zero_optimization"]["stage"] == 3:
        ds_cfg["zero_optimization"]["reduce_bucket_size"] = hidden_size * hidden_size
        ds_cfg["zero_optimization"]["stage3_param_persistence_threshold"] = 10 * hidden_size
        ds_cfg["zero_optimization"]["stage3_prefetch_bucket_size"] = int(0.9 * hidden_size * hidden_size)

    teacher_engine, _, _, _ = deepspeed.initialize(
        model=teacher_model,
        model_parameters=None,
        config=ds_cfg
    )
    teacher_engine.eval()
    return teacher_engine


def patch_model_for_stage1(model, base_model_cfg, cfg):
    """
    Replace `layer.attn` with a wrapper so the teacher’s
    hidden states still drive the rest of the frozen network.
    """
    # Get the correct student attention class dynamically
    student_attn_class = get_student_attention_class(cfg.student_model.name)
    logger.info(f"✅ Using student attention class: {student_attn_class.__name__}")

    # Get the list of layers to keep as full attention from the config.
    # Default to an empty list if not specified.
    keep_full_attention_layers = cfg.student_model.get('keep_full_attention_layers', [])
    if keep_full_attention_layers:
        logger.info(f"⚠️ Will keep the following layers as full-attention: {keep_full_attention_layers}")

    student_name = str(cfg.student_model.get("name", "")).lower()

    for idx, layer in enumerate(model.model.layers):
        # Conditionally skip patching if the layer index is in our keep list.
        if idx in keep_full_attention_layers:
            logger.info(f"  -> Skipping layer {idx}, keeping as full-attention.")
            # Ensure the kept layer is frozen, as it's not being trained in Stage 1.
            for param in layer.attn.parameters():
                param.requires_grad_(False)
            continue

        logger.info(f"  -> Patching layer {idx} with student attention wrapper.")
        teacher_attn = layer.attn
        wrapper = AttentionDistillationWrapper(
            teacher_attn,
            student_attn_class,
            base_model_cfg,
            idx
        )
        layer.attn = wrapper

def build_student_for_stage1(cfg):
    """
    Build and partially freeze the student for stage 1 (attention distillation).
    Typically we load from the base model and selectively unfreeze Q/K/V or
    additional trainable layers.
    """
    base_model_cfg = AutoConfig.from_pretrained(cfg.teacher_model.name)

    base_model_cfg.use_cache = False # important!

    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model.name,
        config=base_model_cfg,
        torch_dtype=torch.bfloat16,
    )

    # Patch each layer with (teacher → wrapper → student)
    patch_model_for_stage1(model, base_model_cfg, cfg)

    if cfg.train.get('gradient_checkpointing', False):
        logger.info("✅ Enabling gradient checkpointing for Stage 1 model")
        model.gradient_checkpointing_enable()

    # Freeze everything that is NOT inside .student_attn.
    for name, p in model.named_parameters():
        p.requires_grad_( ".student_attn." in name )

    tr, tot = count_model_params(model, True), count_model_params(model, False)
    logger.info(f"Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")
    return model


def build_student_for_stage2_and_3(cfg):
    """
    Build the stage 2 student by loading the all-linear checkpoint from stage 1
    and then "restoring" the full-attention teacher layers.
    """
    # Load the config from the Stage 1 (all-linear) checkpoint
    student_config = AutoConfig.from_pretrained(cfg.train.student_init_ckpt)

    target_keep_layers = cfg.student_model.get('keep_full_attention_layers', [])
    student_config.keep_full_attention_layers = list(target_keep_layers)
    student_config.fuse_swiglu = False # to be compatible with DeepSpeed's Zero-3
    logger.info(f"✅ Building Stage 2/3 as HYBRID with keep_layers: {target_keep_layers}")

    # Build the new hybrid model structure (e.g., with empty weights)
    with init_empty_weights():
        student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to_empty(device='cpu')
    student_model = student_model.to(torch.bfloat16)
    
    # Load the weights from the Stage 1 (all-linear) checkpoint
    logger.info(f"➡️ Loading weights from all-linear student: {cfg.train.student_init_ckpt}")
    ckpt_dir = cfg.train.student_init_ckpt
    index_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    safetensors_path = os.path.join(ckpt_dir, 'model.safetensors')
    pytorch_bin_path = os.path.join(ckpt_dir, 'pytorch_model.bin')

    student_sd = {}
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        for shard_file in set(index['weight_map'].values()):
            sf_path = os.path.join(ckpt_dir, shard_file)
            if os.path.exists(sf_path):
                student_sd.update(load_file(sf_path, device="cpu"))
    elif os.path.exists(safetensors_path):
        student_sd = load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_bin_path):
        student_sd = torch.load(pytorch_bin_path, map_location="cpu")
    else:
         raise FileNotFoundError(f"No weights files (.safetensors or .bin) found in {ckpt_dir}")

    student_weights_to_load = {}
    for k, v in student_sd.items():
        is_ssm_layer_key = True
        for layer_idx in target_keep_layers:
            if k.startswith(f"model.layers.{layer_idx}.") and ("self_attn" in k or "attn" in k):
                is_ssm_layer_key = False
                break
        
        if is_ssm_layer_key:
            student_weights_to_load[k] = v

    del student_sd

    # Load all matching weights from the all-linear student (FFNs, embeddings, linear attn layers)
    # This will correctly miss the full-attention layers.
    # this part is just for sanity check!
    load_result_student = student_model.load_state_dict(student_weights_to_load, strict=False)
    logger.info(f"✅ Loaded all-linear student weights (strict=False):")
    logger.info(f"   -> Missing keys (expected): {load_result_student.missing_keys}...")

    # Load the full-attention layers from the Teacher
    logger.info(f"➡️ Restoring hybrid layers from teacher: {cfg.teacher_model.name}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model.name, 
        torch_dtype=torch.bfloat16
    ).to('cpu')
    teacher_sd = teacher_model.state_dict()
    
    teacher_weights_to_load = {}
    for k, v in teacher_sd.items():
        is_hybrid_layer_key = False
        for layer_idx in target_keep_layers:
            if k.startswith(f"model.layers.{layer_idx}.") and ("self_attn" in k or "attn" in k):
                is_hybrid_layer_key = True
                break
        
        if is_hybrid_layer_key:
            teacher_weights_to_load[k] = v

    del teacher_model
    del teacher_sd

    combined_weights_to_load = student_weights_to_load | teacher_weights_to_load
    
    student_model.load_state_dict(combined_weights_to_load)

    for name, p in student_model.named_parameters():
        p.requires_grad = True
        
    if cfg.train.get('gradient_checkpointing', False):
        logger.info("✅ Enabling gradient checkpointing for Stage 2 model")
        student_model.gradient_checkpointing_enable()
        
    tr, tot = count_model_params(student_model, True), count_model_params(student_model, False)
    logger.info(f"[Stage 2] Hybrid Student: Trainable = {tr/1e6:.1f}M | Total = {tot/1e6:.1f}M ({tr/tot:.2%})")

    del student_weights_to_load
    del teacher_weights_to_load
    del combined_weights_to_load
    gc.collect()

    
    return student_model


def build_teacher_for_stage2(cfg):
    """
    Teacher is the base model with full attention. If you want to
    DeepSpeed-shard it, do so. Otherwise, just load it normally.
    """
    teacher_config = AutoConfig.from_pretrained(cfg.teacher_model.name)
    teacher_config.fuse_swiglu = False  # DS ZeRO-3 compat
    # Avoid keeping massive KV caches during 16k passes
    teacher_config.use_cache = bool(cfg.train.get('kd_teacher_use_cache', False))

    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model.name,
        config=teacher_config,
        torch_dtype=torch.bfloat16,
    )
    teacher_model.eval()

    if cfg.train.get('gradient_checkpointing', False):
        logger.info("✅ Enabling gradient checkpointing for teacher model")
        teacher_model.gradient_checkpointing_enable()

    # If you need DS-sharding for the teacher:
    teacher_ds_cfg = "ds_configs/stage_2_teacher.json"
    # Allow CPU offload toggle from yaml (default: cpu)
    offload_mode = str(cfg.train.get('kd_teacher_offload', 'cpu')).lower()
    if offload_mode == 'cpu':
        logger.info("✅ Teacher ZeRO-3 with CPU offload")
    else:
        logger.info("✅ Teacher ZeRO-3 without CPU offload")

    teacher_model = _prepare_teacher_deepspeed(teacher_model, teacher_ds_cfg)
    return teacher_model



def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.teacher_model.name,
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine stage
    #    (We assume user sets cfg.stage = 1, 2, 3)
    stage = cfg.stage

    if stage == 1:
        logger.info("==== Stage 1 (Attention Transfer) ====")
        # Student: from base model
        model = build_student_for_stage1(cfg)
        trainer_class = DistillTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_configs/stage_1.json")

    elif stage == 2:
        logger.info("==== Stage 2 (Logit Distillation) ====")
        # Student: from the checkpoint saved by stage 1
        model = build_student_for_stage2_and_3(cfg)
        # Teacher: base model with full attention
        teacher_model = build_teacher_for_stage2(cfg)
        trainer_class = KDTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_configs/stage_2.json")


    elif stage == 3:
        logger.info("==== Stage 3 (Long-Context Finetuning) ====")
        # Student is the checkpoint saved by stage 2
        model = build_student_for_stage2_and_3(cfg)
        # No teacher model in stage 3
        teacher_model = None
        # Use the standard fine-tuning trainer
        trainer_class = FinetuneTrainer
        ds_config_path = os.path.join(os.getcwd(), "ds_configs/stage_3.json")
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 1, 2, or 3.")


    if os.path.exists(ds_config_path):
        logger.info(f"Using DS config = {ds_config_path}")
    else:
        ds_config_path = None
    
    from data import get_dataloader
    train_loader = get_dataloader(cfg.data.cache_dir, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)

    def get_optimizer(model, config):
        attn_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "attn" in name:
                attn_params.append(param)
                logger.info(f"Attn params: {name}, lr: {config.train.lr_attn}")
            else:
                other_params.append(param)
                logger.info(f"Other params: {name}, lr: {config.train.lr}")
        optimizer_grouped_parameters = [
            {"params": attn_params, "lr": config.train.lr_attn},
            {"params": other_params, "lr": config.train.lr},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), fused=True)
        return optimizer
    
    world_size_env = int(os.environ.get("WORLD_SIZE", "0"))
    num_gpus = world_size_env if world_size_env > 0 else max(1, torch.cuda.device_count())
    seq_len = cfg.train.train_seq_len
    tgt_tok = cfg.train.target_tokens

    # Total micro batch across the whole world per step
    micro_total = cfg.train.micro_batch_size * num_gpus

    # Robust accumulation: ceil to guarantee >= 1
    g_accum = max(1, math.ceil(cfg.train.batch_size / max(1, micro_total)))

    # The actual effective global batch the trainer will use
    effective_global_batch = micro_total * g_accum

    # Derive max_steps from the effective global batch, so target_tokens stays correct
    if tgt_tok:
        max_steps = tgt_tok // (effective_global_batch * seq_len)
        max_steps = max(1, int(max_steps))
    else:
        max_steps = cfg.train.max_steps

    logger.info(f"gradient accumulation steps (computed): {g_accum}")
    logger.info(f"max steps: {max_steps}")
    logger.info(f"desired global batch (cfg.train.batch_size): {cfg.train.batch_size}")
    logger.info(f"effective global batch (micro*world*accum): {effective_global_batch}")
    logger.info(f"per-device micro batch size: {cfg.train.micro_batch_size}")
    logger.info(f"world size (GPUs): {num_gpus}")
    logger.info(f"target tokens: {cfg.train.target_tokens}")
    logger.info(f"train seq len: {cfg.train.train_seq_len}")

    training_args = TrainingArguments(
        per_device_train_batch_size = cfg.train.micro_batch_size,
        gradient_accumulation_steps = g_accum,
        max_steps                   = max_steps,
        bf16                        = True,
        logging_steps               = 10,
        eval_strategy               = "no",
        eval_steps                  = 5000000,
        save_steps                  = 10000,  # good?
        save_total_limit            = 100,
        metric_for_best_model       = "loss",
        greater_is_better           = False,
        output_dir                  = cfg.train.output_dir,
        deepspeed                   = ds_config_path,
        report_to                   = 'wandb',
        gradient_checkpointing      = cfg.train.get('gradient_checkpointing', False),
        learning_rate               = cfg.train.lr,
        lr_scheduler_type           = cfg.train.lr_scheduler_type,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_loader.dataset,
        "eval_dataset": None,
        "optimizers": (get_optimizer(model, cfg), None), # auto infer lr scheduler
        "tokenizer": tokenizer,
    }

    if stage == 1:
        trainer_kwargs["mse_factor"] = 1.0
        trainer = DistillTrainer(**trainer_kwargs)
    elif stage == 2:
        trainer_kwargs["teacher_model"] = teacher_model
        trainer_kwargs["kl_weight"] = 1
        trainer_kwargs["ce_weight"] = 0
        trainer = KDTrainer(**trainer_kwargs)
    elif stage == 3:
        # FinetuneTrainer takes no extra args from this list
        trainer = FinetuneTrainer(**trainer_kwargs)

    if cfg.train.resume_from_checkpoint == "None":
        trainer.train(resume_from_checkpoint=None)
    else:
        trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg_dict = parse_config(args.cfg)
    cfg = OmegaConf.create(cfg_dict)
    main(cfg)

    if dist.is_initialized():
        if dist.get_rank() == 0:
            keep_layers = OmegaConf.to_container(cfg.student_model.keep_full_attention_layers)
            convert_cmd = [
                "python", "convert_ckpt.py",
                "--cfg", args.cfg,
            ]
            subprocess.run(convert_cmd)


            
